#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <ipa_room_segmentation/timer.h>
#include <ipa_room_segmentation/contains.h>
#include <ipa_room_segmentation/features.h>

#include <ipa_room_segmentation/xgboost_classifier.h>
#include <ipa_room_segmentation/wavefront_region_growing.h>

bool XgboostClassifier::segmentMap(const cv::Mat& map_to_be_labeled, cv::Mat& segmented_map, double map_resolution_from_subscription,
  double room_area_factor_lower_limit, double room_area_factor_upper_limit) {
  std::cout << "Start prediction..." << std::endl;

  cv::Mat original_map_to_be_labeled = cv::Mat(map_to_be_labeled.size(), CV_8U);

  std::vector<double> temporary_beams;
  std::vector<float> temporary_features;
  LaserScannerFeatures lsf;

  assert(_booster != NULL && trained_);
  for (int y = 0; y < map_to_be_labeled.rows; y++) {
    for (int x = 0; x < map_to_be_labeled.cols; x++) {
      if (map_to_be_labeled.at<unsigned char>(y, x) != 0) {
        //simulate the beams and features for every position and save it
        raycasting_.raycasting(map_to_be_labeled, cv::Point(x, y), temporary_beams);
        cv::Mat features;
        lsf.get_features(temporary_beams, angles_for_simulation_, cv::Point(x, y), features);
        temporary_features.resize(features.cols);

        Matrix d_features(1, 23);
        d_features << features.at<float>(0, 0), features.at<float>(0, 1), features.at<float>(0, 2), features.at<float>(0, 3), features.at<float>(0, 4)
          , features.at<float>(0, 5), features.at<float>(0, 6), features.at<float>(0, 7), features.at<float>(0, 8), features.at<float>(0, 9)
          , features.at<float>(0, 10), features.at<float>(0, 11), features.at<float>(0, 12), features.at<float>(0, 13), features.at<float>(0, 14)
          , features.at<float>(0, 15), features.at<float>(0, 16), features.at<float>(0, 17), features.at<float>(0, 18), features.at<float>(0, 19)
          , features.at<float>(0, 20), features.at<float>(0, 21), features.at<float>(0, 22);

        Matrix result;
        int ret = predict(d_features, result);
        if (ret != 0) {
          std::cout << "predict error" << std::endl;
        }

        double probability_for_room = result(0, 0);
        double probability_for_hallway = 1.0 - probability_for_room;

        if (probability_for_room > probability_for_hallway) {
          original_map_to_be_labeled.at<unsigned char>(y, x) = 150; //label it as room
        }
        else {
          original_map_to_be_labeled.at<unsigned char>(y, x) = 100; //label it as hallway
        }
      }
    }
  }

  std::cout << "labeled all white pixels: " << std::endl;
  //******************** III. Apply a median filter over the image to smooth the results.***************************
  cv::Mat temporary_map = original_map_to_be_labeled.clone();
  cv::medianBlur(temporary_map, temporary_map, 3);
  std::cout << "blurred image" << std::endl;

  //make regions black, that have been black before
  for (int x = 0; x < original_map_to_be_labeled.rows; x++) {
    for (int y = 0; y < original_map_to_be_labeled.cols; y++) {
      if (original_map_to_be_labeled.at<unsigned char>(x, y) == 0) {
        temporary_map.at<unsigned char>(x, y) = 0;
      }
    }
  }

  cv::Mat blured_image_for_thresholding = temporary_map.clone();

  //*********** IV. Fill the large enough rooms with a random color and split the hallways into smaller regions*********
  std::vector<std::vector<cv::Point> > contours, temporary_contours, saved_room_contours, saved_hallway_contours;
  //hierarchy saves if the contours are hole-contours:
  //hierarchy[{0,1,2,3}]={next contour (same level), previous contour (same level), child contour, parent contour}
  //child-contour = 1 if it has one, = -1 if not, same for parent_contour
  std::vector < cv::Vec4i > hierarchy;

  std::vector < cv::Scalar > already_used_colors; //saving-vector for the already used coloures

  //find the contours, which are labeled as a room
  cv::threshold(temporary_map, temporary_map, 120, 255, cv::THRESH_BINARY); //find rooms (value = 150)
  cv::findContours(temporary_map, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
  cv::drawContours(blured_image_for_thresholding, contours, -1, cv::Scalar(0), cv::FILLED); //make the found regions at the original map black, because they have been looked at already

  //only take rooms that are large enough and that are not a hole-contour
  for (int c = 0; c < contours.size(); c++) {
    if (map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[c]) > room_area_factor_lower_limit
      && hierarchy[c][3] != 1) {
      saved_room_contours.push_back(contours[c]);
    }
  }

  //find the contours, which are labeled as a hallway
  map_to_be_labeled.convertTo(segmented_map, CV_32SC1, 256, 0);       // rescale to 32 int, 255 --> 255*256 = 65280
  temporary_map = blured_image_for_thresholding.clone();

  cv::threshold(temporary_map, temporary_map, 90, 255, cv::THRESH_BINARY); //find hallways (value = 100)
  cv::findContours(temporary_map, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

  //if the hallway-contours are too big split them into smaller regions, also don't take too small regions
  for (int contour_counter = 0; contour_counter < contours.size(); contour_counter++) {
    if (map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[contour_counter]) > room_area_factor_upper_limit) {
      //Generate a black map to draw the hallway-contour in. Then use this map to ckeck if the generated random Points
      // are inside the contour.
      cv::Mat contour_Map = cv::Mat::zeros(temporary_map.rows, temporary_map.cols, CV_8UC1);
      cv::drawContours(contour_Map, contours, contour_counter, cv::Scalar(255), cv::FILLED);
      cv::erode(contour_Map, contour_Map, cv::Mat(), cv::Point(-1, -1), 10);
      //center-counter so enough centers could be found
      int center_counter = 0;
      //saving-vector for watershed centers
      std::vector < cv::Point > temporary_watershed_centers;

      //find enough random watershed centers that are inside the hallway-contour
      do {
        int random_x = rand() % temporary_map.rows;
        int random_y = rand() % temporary_map.cols;

        if (contour_Map.at<unsigned char>(random_y, random_x) == 255) {
          temporary_watershed_centers.push_back(cv::Point(random_x, random_y));
          center_counter++;
        }
      } while (center_counter <= (map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[contour_counter])) / 8);

      cv::Mat temporary_Map_to_wavefront;
      contour_Map.convertTo(temporary_Map_to_wavefront, CV_32SC1, 256, 0);

      //draw the centers as white circles into a black map and give the center-map and the contour-map to the opencv watershed-algorithm
      for (int current_center = 0; current_center < temporary_watershed_centers.size(); current_center++) {
        bool coloured = false;

        do {
          cv::Scalar fill_colour(rand() % 52224 + 13056);

          if (!contains(already_used_colors, fill_colour)) {
            cv::circle(temporary_Map_to_wavefront, temporary_watershed_centers[current_center], 2, fill_colour, cv::FILLED);
            already_used_colors.push_back(fill_colour);
            coloured = true;
          }
        } while (!coloured);
      }

      //make sure all previously black Pixels are still black
      for (int x = 0; x < map_to_be_labeled.rows; x++) {
        for (int y = 0; y < map_to_be_labeled.cols; y++) {
          if (map_to_be_labeled.at<unsigned char>(x, y) == 0) {
            temporary_Map_to_wavefront.at<unsigned char>(x, y) = 0;
          }
        }
      }

      wavefrontRegionGrowing(temporary_Map_to_wavefront);

      //draw the seperated contour into the map, which should be labeled
      for (int row = 0; row < segmented_map.rows; row++) {
        for (int col = 0; col < segmented_map.cols; col++) {
          if (temporary_Map_to_wavefront.at<int>(row, col) != 0) {
            segmented_map.at<int>(row, col) = temporary_Map_to_wavefront.at<int>(row, col);
          }
        }
      }
    }
    else
      if (map_resolution_from_subscription * map_resolution_from_subscription * cv::contourArea(contours[contour_counter])
        > room_area_factor_lower_limit) {
        saved_hallway_contours.push_back(contours[contour_counter]);
      }
  }

  std::cout << "finished too big hallway contours" << std::endl;

  //draw every room and lasting hallway contour with a random colour into the map
  for (int room = 0; room < saved_room_contours.size(); room++) {
    bool coloured = false;

    do {
      cv::Scalar fill_colour(rand() % 52224 + 13056);

      if (!contains(already_used_colors, fill_colour)) {
        cv::drawContours(segmented_map, saved_room_contours, room, fill_colour, cv::FILLED);
        already_used_colors.push_back(fill_colour);
        coloured = true;
      }
    } while (!coloured);
  }

  std::cout << "finished room contours" << std::endl;

  for (int hallway = 0; hallway < saved_hallway_contours.size(); hallway++) {
    bool coloured = false;
    int loop_counter = 0; //loop-counter to exit the loop if it gets a infite loop

    do {
      loop_counter++;
      cv::Scalar fill_colour(rand() % 52224 + 13056);

      if (!contains(already_used_colors, fill_colour) || loop_counter > 250) {
        cv::drawContours(segmented_map, saved_hallway_contours, hallway, fill_colour, cv::FILLED);
        already_used_colors.push_back(fill_colour);
        coloured = true;
      }
    } while (!coloured);
  }

  std::cout << "finished small hallway contours" << std::endl;
  //spread the coloured regions to regions, which were too small and aren't drawn into the map
  wavefrontRegionGrowing(segmented_map);

  //make sure previously black pixels are still black
  for (int v = 0; v < map_to_be_labeled.rows; ++v) {
    for (int u = 0; u < map_to_be_labeled.cols; ++u) {
      if (map_to_be_labeled.at<unsigned char>(v, u) == 0) {
        segmented_map.at<int>(v, u) = 0;
      }
    }
  }

  std::cout << "Finished Labeling the map." << std::endl;
  return trained_;
}
