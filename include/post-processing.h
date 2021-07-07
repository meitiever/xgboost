#pragma once

#include <list>
#include <ctime>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <ipa_room_segmentation/meanshift2d.h>
#include <ipa_room_segmentation/features.h>
#include <ros/ipa_building_msgs/room-information.h>
#include <ros/ipa_building_msgs/map-segmentation-results.h>
#include <ros/geometry_msgs/point32.h>

class PostProcessor {
protected:

  //converter-> Pixel to meter for X coordinate
  double convert_pixel_to_meter_for_x_coordinate(const int pixel_valued_object_x, const float map_resolution, const cv::Point2d map_origin)
  {
    double meter_value_obj_x = (pixel_valued_object_x * map_resolution) + map_origin.x;
    return meter_value_obj_x;
  }
  //converter-> Pixel to meter for Y coordinate
  double convert_pixel_to_meter_for_y_coordinate(int pixel_valued_object_y, const float map_resolution, const cv::Point2d map_origin)
  {
    double meter_value_obj_y = (pixel_valued_object_y * map_resolution) + map_origin.y;
    return meter_value_obj_y;
  }

public:

  PostProcessor() { } // default constructor

  //training-method for the classifier
  void Process(const cv::Mat& segmented_map, std::vector<cv::Point>& doorway_points_, geometry_msgs::Pose map_origin_, cv::Mat& out_map, ipa_building_msgs::MapSegmentationResult& result, double robot_radius = 0.3, double map_resolution = 0.05) {
    std::cout << "********Post Process Segmented Map************" << std::endl;

    // get the min/max-values and the room-centers
    // compute room label codebook
    std::map<int, size_t> label_vector_index_codebook; // maps each room label to a position in the rooms vector
    size_t vector_index = 0;
    bool return_format_in_meter = false;
    bool return_format_in_pixel = true;
    const cv::Point2d map_origin(map_origin_.position.x, map_origin_.position.y);

    for (int v = 0; v < segmented_map.rows; ++v) {
      for (int u = 0; u < segmented_map.cols; ++u) {
        const int label = segmented_map.at<int>(v, u);

        if (label > 0 && label < 65280) { // do not count walls/obstacles or free space as label
          if (label_vector_index_codebook.find(label) == label_vector_index_codebook.end()) {
            label_vector_index_codebook[label] = vector_index;
            vector_index++;
          }
        }
      }
    }

    //min/max y/x-values vector for each room. Initialized with extreme values
    std::vector<int> min_x_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_x_value_of_the_room(label_vector_index_codebook.size(), 0);
    std::vector<int> min_y_value_of_the_room(label_vector_index_codebook.size(), 100000000);
    std::vector<int> max_y_value_of_the_room(label_vector_index_codebook.size(), 0);
    //vector of the central Point for each room, initially filled with Points out of the map
    std::vector<int> room_centers_x_values(label_vector_index_codebook.size(), -1);
    std::vector<int> room_centers_y_values(label_vector_index_codebook.size(), -1);

    //***********************Find min/max x and y coordinate and center of each found room********************
    //check y/x-value for every Pixel and make the larger/smaller value to the current value of the room
    for (int y = 0; y < segmented_map.rows; ++y) {
      for (int x = 0; x < segmented_map.cols; ++x) {
        const int label = segmented_map.at<int>(y, x);

        if (label > 0 && label < 65280) { //if Pixel is white or black it is no room --> doesn't need to be checked
          const int index = label_vector_index_codebook[label];
          min_x_value_of_the_room[index] = std::min(x, min_x_value_of_the_room[index]);
          max_x_value_of_the_room[index] = std::max(x, max_x_value_of_the_room[index]);
          max_y_value_of_the_room[index] = std::max(y, max_y_value_of_the_room[index]);
          min_y_value_of_the_room[index] = std::min(y, min_y_value_of_the_room[index]);
        }
      }
    }

    //get centers for each room
    //  for (size_t idx = 0; idx < room_centers_x_values.size(); ++idx)
    //  {
    //      if (max_x_value_of_the_room[idx] != 0 && max_y_value_of_the_room[idx] != 0 && min_x_value_of_the_room[idx] != 100000000 && min_y_value_of_the_room[idx] != 100000000)
    //      {
    //          room_centers_x_values[idx] = (min_x_value_of_the_room[idx] + max_x_value_of_the_room[idx]) / 2;
    //          room_centers_y_values[idx] = (min_y_value_of_the_room[idx] + max_y_value_of_the_room[idx]) / 2;
    //          cv::circle(segmented_map, cv::Point(room_centers_x_values[idx], room_centers_y_values[idx]), 2, cv::Scalar(200*256), cv::FILLED);
    //      }
    //  }
    // use distance transform and mean shift to find good room centers that are reachable by the robot
    // first check whether a robot radius shall be applied to obstacles in order to exclude room center points that are not reachable by the robot
    cv::Mat segmented_map_copy = segmented_map;
    cv::Mat connection_to_other_rooms = cv::Mat::zeros(segmented_map.rows, segmented_map.cols, CV_8UC1);    // stores for each pixel whether a path to another rooms exists for a robot of size robot_radius

    if (robot_radius > 0.0) {
      // consider robot radius for exclusion of non-reachable points
      segmented_map_copy = segmented_map.clone();
      cv::Mat map_8u, eroded_map;
      segmented_map_copy.convertTo(map_8u, CV_8UC1, 1., 0.);
      int number_of_erosions = (robot_radius / map_resolution);
      cv::erode(map_8u, eroded_map, cv::Mat(), cv::Point(-1, -1), number_of_erosions);

      for (int v = 0; v < segmented_map_copy.rows; ++v)
        for (int u = 0; u < segmented_map_copy.cols; ++u)
          if (eroded_map.at<uchar>(v, u) == 0)
            segmented_map_copy.at<int>(v, u) = 0;

      // compute connectivity of remaining accessible room cells to other rooms
      bool stop = false;

      while (stop == false) {
        stop = true;

        for (int v = 1; v < segmented_map_copy.rows - 1; ++v) {
          for (int u = 1; u < segmented_map_copy.cols - 1; ++u) {
            // skip already identified cells
            if (connection_to_other_rooms.at<uchar>(v, u) != 0)
              continue;

            // only consider cells labeled as a room
            const int label = segmented_map_copy.at<int>(v, u);
            if (label <= 0 || label >= 65280)
              continue;

            for (int dv = -1; dv <= 1; ++dv) {
              for (int du = -1; du <= 1; ++du) {
                if (dv == 0 && du == 0)
                  continue;

                const int neighbor_label = segmented_map_copy.at<int>(v + dv, u + du);

                if (neighbor_label > 0 && neighbor_label < 65280 && (neighbor_label != label || (neighbor_label == label && connection_to_other_rooms.at<uchar>(v + dv, u + du) == 255))) {
                  // either the room cell has a direct border to a different room or the room cell has a neighbor from the same room label with a connecting path to another room
                  connection_to_other_rooms.at<uchar>(v, u) = 255;
                  stop = false;
                }
              }
            }
          }
        }
      }
    }

    // compute the room centers
    MeanShift2D ms;

    for (std::map<int, size_t>::iterator it = label_vector_index_codebook.begin(); it != label_vector_index_codebook.end(); ++it) {
      int trial = 1;  // use robot_radius to avoid room centers that are not accessible by a robot with a given radius

      if (robot_radius <= 0.)
        trial = 2;

      for (; trial <= 2; ++trial) {
        // compute distance transform for each room on the room cells that have some connection to another room (trial 1) or just on all cells of that room (trial 2)
        const int label = it->first;
        int number_room_pixels = 0;
        cv::Mat room = cv::Mat::zeros(segmented_map_copy.rows, segmented_map_copy.cols, CV_8UC1);

        for (int v = 0; v < segmented_map_copy.rows; ++v)
          for (int u = 0; u < segmented_map_copy.cols; ++u)
            if (segmented_map_copy.at<int>(v, u) == label && (trial == 2 || connection_to_other_rooms.at<uchar>(v, u) == 255)) {
              room.at<uchar>(v, u) = 255;
              ++number_room_pixels;
            }

        if (number_room_pixels == 0)
          continue;

        cv::Mat distance_map; //variable for the distance-transformed map, type: CV_32FC1
        cv::distanceTransform(room, distance_map, cv::DIST_L2, 5);
        // find point set with largest distance to obstacles
        double min_val = 0., max_val = 0.;
        cv::minMaxLoc(distance_map, &min_val, &max_val);
        std::vector<cv::Vec2d> room_cells;

        for (int v = 0; v < distance_map.rows; ++v)
          for (int u = 0; u < distance_map.cols; ++u)
            if (distance_map.at<float>(v, u) > max_val * 0.95f)
              room_cells.push_back(cv::Vec2d(u, v));

        if (room_cells.size() == 0)
          continue;

        // use meanshift to find the modes in that set
        cv::Vec2d room_center = ms.findRoomCenter(room, room_cells, map_resolution);
        const int index = it->second;
        room_centers_x_values[index] = room_center[0];
        room_centers_y_values[index] = room_center[1];

        if (room_cells.size() > 0)
          break;
      }
    }

    // convert the segmented map into an indexed map which labels the segments with consecutive numbers (instead of arbitrary unordered labels in segmented map)
    cv::Mat indexed_map = segmented_map.clone();

    for (int y = 0; y < segmented_map.rows; ++y) {
      for (int x = 0; x < segmented_map.cols; ++x) {
        const int label = segmented_map.at<int>(y, x);

        if (label > 0 && label < 65280)
        {
          indexed_map.at<int>(y, x) = label_vector_index_codebook[label] + 1;
        } //start value from 1 --> 0 is reserved for obstacles
      }
    }

    //****************publish the results**********************//
    out_map = indexed_map.clone();

    //setting value to the action msgs to publish
    result.map_resolution = map_resolution;
    result.map_origin = map_origin_;

    //setting massages in pixel value
    result.room_information_in_pixel.clear();

    if (return_format_in_pixel) {
      std::vector<ipa_building_msgs::RoomInformation> room_information(room_centers_x_values.size());

      for (size_t i = 0; i < room_centers_x_values.size(); ++i) {
        room_information[i].room_center.x = room_centers_x_values[i];
        room_information[i].room_center.y = room_centers_y_values[i];
        room_information[i].room_min_max.points.resize(2);
        room_information[i].room_min_max.points[0].x = min_x_value_of_the_room[i];
        room_information[i].room_min_max.points[0].y = min_y_value_of_the_room[i];
        room_information[i].room_min_max.points[1].x = max_x_value_of_the_room[i];
        room_information[i].room_min_max.points[1].y = max_y_value_of_the_room[i];
      }

      result.room_information_in_pixel = room_information;

      // returning doorway points if the vector is not empty
      if (doorway_points_.empty() == false) {
        std::vector<geometry_msgs::Point32> found_doorway_points(doorway_points_.size());

        for (size_t i = 0; i < doorway_points_.size(); ++i) {
          found_doorway_points[i].x = doorway_points_[i].x;
          found_doorway_points[i].y = doorway_points_[i].y;
        }

        doorway_points_.clear();

        result.doorway_points = found_doorway_points;
      }
    }

    //setting messages in meter
    result.room_information_in_meter.clear();

    if (return_format_in_meter) {
      std::vector<ipa_building_msgs::RoomInformation> room_information(room_centers_x_values.size());

      for (size_t i = 0; i < room_centers_x_values.size(); ++i) {
        room_information[i].room_center.x = convert_pixel_to_meter_for_x_coordinate(room_centers_x_values[i], map_resolution, map_origin);
        room_information[i].room_center.y = convert_pixel_to_meter_for_y_coordinate(room_centers_y_values[i], map_resolution, map_origin);
        room_information[i].room_min_max.points.resize(2);
        room_information[i].room_min_max.points[0].x = convert_pixel_to_meter_for_x_coordinate(min_x_value_of_the_room[i], map_resolution, map_origin);
        room_information[i].room_min_max.points[0].y = convert_pixel_to_meter_for_y_coordinate(min_y_value_of_the_room[i], map_resolution, map_origin);
        room_information[i].room_min_max.points[1].x = convert_pixel_to_meter_for_x_coordinate(max_x_value_of_the_room[i], map_resolution, map_origin);
        room_information[i].room_min_max.points[1].y = convert_pixel_to_meter_for_y_coordinate(max_y_value_of_the_room[i], map_resolution, map_origin);
      }

      result.room_information_in_meter = room_information;

      // returning doorway points if the vector is not empty
      if (doorway_points_.empty() == false) {
        std::vector<geometry_msgs::Point32> found_doorway_points(doorway_points_.size());

        for (size_t i = 0; i < doorway_points_.size(); ++i) {
          found_doorway_points[i].x = convert_pixel_to_meter_for_x_coordinate(doorway_points_[i].x, map_resolution, map_origin);;
          found_doorway_points[i].y = convert_pixel_to_meter_for_y_coordinate(doorway_points_[i].y, map_resolution, map_origin);
        }

        doorway_points_.clear();

        result.doorway_points = found_doorway_points;
      }
    }

    //publish result
    std::cout << "********Map segmentation finished************" << std::endl;
  }
};
