#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>

#include <ctime>
#include <stdlib.h>

#include <ipa_room_segmentation/features.h>
#include <ipa_room_segmentation/raycasting.h>

class PreProcessor {
public:

  PreProcessor() { } // default constructor

  //training-method for the classifier
  cv::Mat Process(const cv::Mat& img, cv::Mat& out) {
    std::cout << "********Pre Process Map************" << std::endl;
    out = img.clone();
    if (img.channels() == 3) {
      for (int i = 0; i < img.size().height; i++) {
        for (int j = 0; j < img.size().width; j++) {
          cv::Vec3b px = img.at<cv::Vec3b>(i, j);
          int r = px[0];
          int g = px[1];
          int b = px[2];
          if ((r == 0 && g == 255 && b == 25) || (r == 128 && g == 255 && b == 140) || (r == 191 && g == 255 && b == 191)
            || (r == 64 && g == 255 && b == 82) || (r == 191 && g == 255 && b == 197) || (r == 122 && g == 250 && b == 135)
            || (r == 61 && g == 252 && b == 80) || (r == 124 && g == 251 && b == 136) || (r == 125 && g == 252 && b == 137)
            || (r == 124 && g == 251 && b == 136) || (r == 124 && g == 251 && b == 139) || (r == 126 && g == 254 && b == 139)
            || (r == 186 && g == 250 && b == 192) || (r == 187 && g == 251 && b == 194) || (r == 62 && g == 253 && b == 81)
            || (r == 116 && g == 244 && b == 129) || (r == 41 && g == 232 && b == 60) || (r == 60 && g == 251 && b == 79)) {
            out.at<cv::Vec3b>(i, j) = white;
          }
        }
      }
    }

    //make non-white pixels black
    for (int y = 0; y < img.rows; y++) {
      for (int x = 0; x < img.cols; x++) {
        //find not reachable regions and make them black
        if (img.at<unsigned char>(y, x) >= 250)
        {
          out.at<unsigned char>(y, x) = 255;
        }
        else
        {
          out.at<unsigned char>(y, x) = 0;
        }
      }
    }

    std::vector<double> angles_for_simulation_; // angle-vector used to calculate the features for this algorithm
    std::vector<double> temporary_beams;
    LaserScannerRaycasting raycasting_;

    for (double angle = 0; angle < 360; angle++) {
      angles_for_simulation_.push_back(angle);
    }

    int num_features = 0;
    for (int y = 0; y < img.rows; y++) {
      for (int x = 0; x < img.cols; x++) {
        if (img.at<unsigned char>(y, x) != 0) {
          num_features++;
        }
      }
    }

    cv::Mat features = cv::Mat(num_features, 23, CV_32FC1);

    num_features = 0;
#pragma omp parallel for
    for (int y = 0; y < img.rows; y++) {
      LaserScannerFeatures lsf;

      for (int x = 0; x < img.cols; x++) {
        if (img.at<unsigned char>(y, x) != 0) {
          std::vector<double> temporary_beams;
          //simulate the beams and features for every position and save it
          raycasting_.raycasting(img, cv::Point(x, y), temporary_beams);
          cv::Mat feature;
          lsf.get_features(temporary_beams, angles_for_simulation_, cv::Point(x, y), feature);
          // write features
          for (int i = 0; i < feature.cols; ++i)
            features.at<float>(num_features, i) = feature.at<float>(0, i);
          num_features++;
        }
      }
    }

    return features;
  }
};
