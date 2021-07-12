/*!
 * Copyright 2014-2020 by Contributors
 * \file cli_main.cc
 * \brief The command line interface program of xgboost.
 *  This file is not included in dynamic library.
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <dmlc/timer.h>

#include <xgboost/learner.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/parameter.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include <iomanip>
#include <ctime>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/io.h"
#include "common/common.h"
#include "common/config.h"
#include "common/version.h"
#include "c_api/c_api_utils.h"

#include <utils.h>
#include <pre-processing.h>
#include <post-processing.h>
#include <boost/filesystem.hpp>

#include <ros/ipa_building_msgs/map-segmentation-results.h>
#include <ipa_room_segmentation/evaluation_segmentation.h>
#include <ipa_room_segmentation/distance_segmentation.h>
#include <ipa_room_segmentation/morphological_segmentation.h>
#include <ipa_room_segmentation/voronoi_segmentation.h>
#include <ipa_room_segmentation/adaboost_classifier.h>
#include <ipa_room_segmentation/xgboost_classifier.h>
#include <ipa_room_segmentation/voronoi_random_field_segmentation.h>

using namespace cv;
using namespace std;

static void help(char** argv)
{
  cout << "This sample program demonstrates the room segmentation algorithm\n"
    << "Call:\n"
    << argv[0] << " --help h ? |      | print help message" << endl
    << "--root     | D:\\Github\\Tools\\xgboost\\ | root path of file" << endl
    << "--train_semantic | false | train adaboost classifier." << endl
    << "--train_vrf      | false | train vrf classifier." << endl
    << "--map_resolution | 0.015 | map resolution." << endl
    << "--preprocess     | false | do pre process." << endl
    << "--draw_obstacles | false | draw obstacles on segmented map." << endl
    << "--save_to_csv    | false | parse features to xgboost libsvm format." << endl
    << "--xgboost_model  | files\\classifier_models\\semantic_room_xgboost_r100.model | xgboost config files path semantic_room_xgboost_r100.model";
}

int main(int argc, char* argv[]) {
  cv::CommandLineParser parser(argc, argv,
    "{help h ? |      | help message}"
    "{root     | D:\\Github\\Tools\\xgboost\\ | root path of file }"
    "{train_semantic | false | train adaboost classifier. }"
    "{train_vrf      | false | train vrf classifier. }"
    "{map_resolution | 0.015 | map resolution. }"
    "{draw_obstacles | false | draw obstacles on segmented map. }"
    "{preprocess     | false | parse features to xgboost libsvm format. }"
    "{save_to_csv    | false | parse features to xgboost libsvm format. }"
    "{xgboost_model  | files\\classifier_models\\semantic_room_xgboost_r100.model | xgboost config files path semantic_room_xgboost_r100.model }"
  );

  if (parser.has("help"))
  {
    help(argv);
    return 0;
  }

  double map_resolution = parser.get<double>("map_resolution");
  std::cout << "parameter map_resolution set to " << map_resolution << std::endl;

  bool save_to_csv = parser.get<bool>("save_to_csv");
  std::cout << "parameter save_to_csv set to " << (save_to_csv ? "true" : "false") << std::endl;

  bool draw_obstacles = parser.get<bool>("draw_obstacles");
  std::cout << "parameter draw_obstacles set to " << (draw_obstacles ? "true" : "false") << std::endl;

  bool preprocess = parser.get<bool>("preprocess");
  std::cout << "parameter preprocess set to " << (preprocess ? "true" : "false") << std::endl;

  std::string package_path = parser.get<std::string>("root");

  if (!boost::filesystem::exists(package_path))
  {
    cerr << "Path : " << package_path << " does not exists." << endl;
    exit(-1);
  }

  std::string path = parser.get<std::string>("xgboost_model");
  std::string xgboost_path = package_path + path;

  if (!boost::filesystem::is_regular_file(xgboost_path))
  {
    cerr << "File : " << xgboost_path << " does not exists." << endl;
    exit(-1);
  }

  std::vector<std::string> map_names;
  map_names.push_back("sineva01");
  map_names.push_back("sineva02");
  map_names.push_back("sineva03");
  map_names.push_back("sineva04");
  map_names.push_back("sineva05");
  map_names.push_back("sineva06");
  map_names.push_back("sineva07");
  map_names.push_back("sineva08");
  map_names.push_back("sineva09");
  map_names.push_back("sineva10");
  map_names.push_back("sineva11");
  map_names.push_back("sineva12");
  map_names.push_back("sineva13");
  map_names.push_back("sineva14");
  map_names.push_back("sineva15");
  map_names.push_back("sineva16");
  map_names.push_back("sineva17");
  map_names.push_back("sineva18");
  map_names.push_back("sineva19");
  map_names.push_back("sineva20");
  map_names.push_back("sineva21");
  map_names.push_back("sineva22");
  map_names.push_back("sineva23");
  map_names.push_back("sineva24");
  map_names.push_back("sineva25");
  map_names.push_back("sineva26");
  map_names.push_back("sineva27");
  map_names.push_back("sineva28");
  map_names.push_back("sineva29");
  map_names.push_back("sineva30");
  map_names.push_back("sineva31");
  map_names.push_back("sineva32");
  map_names.push_back("lab_ipa");
  map_names.push_back("lab_c_scan");
  map_names.push_back("Freiburg52_scan");
  map_names.push_back("Freiburg79_scan");
  map_names.push_back("lab_b_scan");
  map_names.push_back("lab_intel");
  map_names.push_back("Freiburg101_scan");
  map_names.push_back("lab_d_scan");
  map_names.push_back("lab_f_scan");
  map_names.push_back("lab_a_scan");
  map_names.push_back("NLB");
  map_names.push_back("office_a");
  map_names.push_back("office_b");
  map_names.push_back("office_c");
  map_names.push_back("office_d");
  map_names.push_back("office_e");
  map_names.push_back("office_f");
  map_names.push_back("office_g");
  map_names.push_back("office_h");
  map_names.push_back("office_i");
  map_names.push_back("lab_ipa_furnitures");
  map_names.push_back("lab_c_scan_furnitures");
  map_names.push_back("Freiburg52_scan_furnitures");
  map_names.push_back("Freiburg79_scan_furnitures");
  map_names.push_back("lab_b_scan_furnitures");
  map_names.push_back("lab_intel_furnitures");
  map_names.push_back("Freiburg101_scan_furnitures");
  map_names.push_back("lab_d_scan_furnitures");
  map_names.push_back("lab_f_scan_furnitures");
  map_names.push_back("lab_a_scan_furnitures");
  map_names.push_back("NLB_furnitures");
  map_names.push_back("office_a_furnitures");
  map_names.push_back("office_b_furnitures");
  map_names.push_back("office_c_furnitures");
  map_names.push_back("office_d_furnitures");
  map_names.push_back("office_e_furnitures");
  map_names.push_back("office_f_furnitures");
  map_names.push_back("office_g_furnitures");
  map_names.push_back("office_h_furnitures");
  map_names.push_back("office_i_furnitures");

  std::vector<uint> possible_labels(3); // vector that stores the possible labels that are drawn in the training maps. Order: room - hallway - doorway
  possible_labels[0] = 77;
  possible_labels[1] = 115;
  possible_labels[2] = 179;

  std::string map_path = package_path + "files\\test_maps\\";
  std::string segmented_map_path = package_path + "files\\segmented_maps\\";

  // strings that stores the path to the saving files
  std::string conditional_weights_path = package_path + "files\\classifier_models\\conditional_field_weights.txt";
  std::string boost_file_path = package_path + "files\\classifier_models\\";

  // optimal result saving path
  std::string conditional_weights_optimal_path = package_path + "files\\classifier_models\\vrf_conditional_field_weights.txt";
  std::string boost_file_optimal_path = package_path + "files\\classifier_models\\";

  bool train_semantic_, train_vrf_;
  train_semantic_ = parser.get<bool>("train_semantic");
  train_vrf_ = parser.get<bool>("train_vrf");

  std::cout << "parameter train_semantic_ set to " << (train_semantic_ ? "true" : "false") << std::endl;
  std::cout << "parameter train_vrf_ set to " << (train_vrf_ ? "true" : "false") << std::endl;

  if (train_vrf_) {
    std::cout << "parameter train_vrf set to true." << std::endl;

    // load the training maps
    cv::Mat training_map;
    std::vector<cv::Mat> training_maps;
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_Fr52.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_Fr101.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_intel.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_d_furniture.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_ipa.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_NLB_furniture.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_office_e.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_office_h.png", 0);
    training_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files\\training_maps\\voronoi_random_field_training\\training_maps\\training_lab_c_furnitures.png", 0);
    training_maps.push_back(training_map);
    // load the voronoi maps
    std::vector<cv::Mat> voronoi_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/Fr52_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/Fr101_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_intel_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_d_furnitures_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_ipa_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/NLB_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/office_e_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/office_h_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_maps/lab_c_furnitures_voronoi.png", 0);
    voronoi_maps.push_back(training_map);
    // load the voronoi-nodes maps
    std::vector<cv::Mat> voronoi_node_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/Fr52_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/Fr101_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_intel_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_d_furnitures_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_ipa_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/NLB_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/office_e_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/office_h_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/voronoi_node_maps/lab_c_furnitures_voronoi_nodes.png", 0);
    voronoi_node_maps.push_back(training_map);
    // load the original maps
    std::vector<cv::Mat> original_maps;
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/Fr52_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/Fr101_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_intel_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_d_furnitures_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_ipa_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/NLB_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/office_e_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/office_h_original.png", 0);
    original_maps.push_back(training_map);
    training_map = cv::imread(package_path + "files/training_maps/voronoi_random_field_training/original_maps/lab_c_furnitures_original.png", 0);
    original_maps.push_back(training_map);

    std::cout << "voronoi_random_field_training maps:" << "training maps:" << training_maps.size() << " voronoi maps:" << voronoi_maps.size() << " voronoi node maps:" << voronoi_node_maps.size() << " original maps:" << original_maps.size() << std::endl;
  }

  // list of files containing maps with room labels for training the semantic segmentation
  std::vector<std::string> semantic_training_maps_room_file_list_;
  // list of files containing maps with hallway labels for training the semantic segmentation
  std::vector<std::string> semantic_training_maps_hallway_file_list_;

  if (train_semantic_) {
    std::cout << "parameter train_semantic set to true." << std::endl;
    std::vector<std::string> fileNames;
    if (boost::filesystem::exists(package_path + "files\\training_maps\\sineva\\")) {
      if (getFileNames(package_path + "files\\training_maps\\sineva\\", fileNames) < 1) {
        std::cout << package_path + "files\\training_maps\\sineva\\" << " does not exists or empty." << std::endl;
        return 0;
      }
      for (auto item : fileNames) {
        if (item.find("room_train") != std::string::npos)
          semantic_training_maps_room_file_list_.push_back(item);
        if (item.find("hallway_train") != std::string::npos)
          semantic_training_maps_hallway_file_list_.push_back(item);
      }
    }

    AdaboostClassifier semantic_segmentation;
    const std::string classifier_default_path = package_path + "files\\classifier_models\\";
    const std::string classifier_path = package_path + "files\\classifier_models\\";

    for (size_t i = 0; i < semantic_training_maps_room_file_list_.size(); ++i)
      std::cout << semantic_training_maps_room_file_list_[i] << std::endl;

    for (size_t i = 0; i < semantic_training_maps_hallway_file_list_.size(); ++i)
      std::cout << semantic_training_maps_hallway_file_list_[i] << std::endl;

    std::cout << "You have chosen to train an adaboost classifier for the semantic segmentation method." << std::endl;

    // load the training maps, change to your maps when you want to train different ones
    std::vector<cv::Mat> room_training_maps;

    for (size_t i = 0; i < semantic_training_maps_room_file_list_.size(); ++i) {
      cv::Mat training_map = cv::imread(semantic_training_maps_room_file_list_[i], 0);
      room_training_maps.push_back(training_map);
    }

    std::vector<cv::Mat> hallway_training_maps;

    for (size_t i = 0; i < semantic_training_maps_hallway_file_list_.size(); ++i) {
      cv::Mat training_map = cv::imread(semantic_training_maps_hallway_file_list_[i], 0);
      hallway_training_maps.push_back(training_map);
    }

    //train the algorithm
    semantic_segmentation.trainClassifiers(room_training_maps, hallway_training_maps, classifier_path, save_to_csv);
    std::exit(0);
  }

  std::vector<cv::Point> door_points;
  std::vector<cv::Point> doorway_points_; // vector that saves the found doorway points, when using the 5th algorithm (vrf)
  geometry_msgs::Pose map_origin;

  PreProcessor pre;
  std::vector<std::string> segmentation_names;
  //segmentation_names.push_back("1morphological");
  segmentation_names.push_back("2distance");
  //segmentation_names.push_back("3voronoi");
  //segmentation_names.push_back("4semantic");
  //segmentation_names.push_back("5vrf");
  //segmentation_names.push_back("6xgboost");

  std::vector<cv::Mat> results(segmentation_names.size());
  for (size_t i = 0; i < segmentation_names.size(); ++i)
    results[i] = cv::Mat::zeros(26, map_names.size(), CV_64FC1);

  // loop through map files
  for (size_t image_index = 0; image_index < map_names.size(); ++image_index) {
    //define vectors to save the parameters
    std::vector<int> segments_number_vector(segmentation_names.size());
    std::vector<double> av_area_vector(segmentation_names.size()), max_area_vector(segmentation_names.size()), min_area_vector(segmentation_names.size()), dev_area_vector(segmentation_names.size());
    std::vector<double> av_per_vector(segmentation_names.size()), max_per_vector(segmentation_names.size()), min_per_vector(segmentation_names.size()), dev_per_vector(segmentation_names.size());
    std::vector<double> av_compactness_vector(segmentation_names.size()), max_compactness_vector(segmentation_names.size()), min_compactness_vector(segmentation_names.size()), dev_compactness_vector(segmentation_names.size());
    std::vector<double> av_bb_vector(segmentation_names.size()), max_bb_vector(segmentation_names.size()), min_bb_vector(segmentation_names.size()), dev_bb_vector(segmentation_names.size());
    std::vector<double> av_quo_vector(segmentation_names.size()), max_quo_vector(segmentation_names.size()), min_quo_vector(segmentation_names.size()), dev_quo_vector(segmentation_names.size());
    std::vector<bool> reachable(segmentation_names.size());

    //load map
    std::string map_name = map_names[image_index];
    std::string image_filename = map_path + map_name + ".png";
    std::cout << "load test map: " << image_filename << std::endl;
    cv::Mat map = cv::imread(image_filename.c_str(), 0);
    cv::Mat original_img;

    cv::Mat features_to_classify;
    if (preprocess) {
      //pre-process the image.
      features_to_classify = pre.Process(map, original_img);
      std::cout << "pre-process done and test map: " << image_filename << " has " << features_to_classify.size().height << std::endl;
    }
    else
      original_img = map.clone();

    //segment the given map
    cv::Mat segmented_map;

    //calculate parameters for each segmentation and save it
    for (size_t segmentation_index = 0; segmentation_index < segmentation_names.size(); ++segmentation_index) {
      std::cout << "Evaluating image '" << map_name << "' with segmentation method " << segmentation_names[segmentation_index] << std::endl;

      const int room_segmentation_algorithm = segmentationNameToNumber(segmentation_names[segmentation_index]);

      if (room_segmentation_algorithm == 1) { //morpho
        double room_lower_limit_morphological_ = 0.8;
        double room_upper_limit_morphological_ = 47.0;
        std::cout << "You have chosen the morphological segmentation." << std::endl;
        MorphologicalSegmentation morphological_segmentation; //morphological segmentation method
        morphological_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_morphological_, room_upper_limit_morphological_, draw_obstacles);
      }

      if (room_segmentation_algorithm == 2) { //distance
        double room_lower_limit_distance_ = 0.35;
        double room_upper_limit_distance_ = 163.0;
        std::cout << "You have chosen the distance segmentation." << std::endl;
        DistanceSegmentation distance_segmentation; //distance segmentation method
        distance_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_distance_, room_upper_limit_distance_, draw_obstacles);
      }

      if (room_segmentation_algorithm == 3) { //voronoi
        double room_lower_limit_voronoi_ = 0.1;
        double room_upper_limit_voronoi_ = 1000000.;
        double voronoi_neighborhood_index_ = 280;
        double max_iterations_ = 150;
        double min_critical_point_distance_factor_ = 0.5;
        double max_area_for_merging_ = 12.5;
        std::cout << "You have chosen the Voronoi segmentation" << std::endl;
        VoronoiSegmentation voronoi_segmentation; //voronoi segmentation method
        voronoi_segmentation.segmentMap(original_img, segmented_map, map_resolution, room_lower_limit_voronoi_, room_upper_limit_voronoi_,
          voronoi_neighborhood_index_, max_iterations_, min_critical_point_distance_factor_, max_area_for_merging_, draw_obstacles);
      }

      if (room_segmentation_algorithm == 4) { //semantic
        std::cout << "You have chosen the semantic segmentation." << std::endl;
        double room_lower_limit_semantic_ = 1.0;
        double room_upper_limit_semantic_ = 1000000.;
        AdaboostClassifier semantic_segmentation; //semantic segmentation method
        const std::string classifier_path = package_path + "files\\classifier_models\\";
        semantic_segmentation.segmentMap(original_img, features_to_classify, segmented_map, map_resolution, room_lower_limit_semantic_, room_upper_limit_semantic_,
          classifier_path);
      }

      if (room_segmentation_algorithm == 5) { //voronoi random field
        double room_area_lower_limit_voronoi_random = 1.53;
        double room_area_upper_limit_voronoi_random = 1000000.;
        double max_iterations = 150;
        double voronoi_random_field_epsilon_for_neighborhood = 7;
        double min_neighborhood_size = 5;
        double min_voronoi_random_field_node_distance = 7.0;
        double max_voronoi_random_field_inference_iterations = 9000;
        double max_area_for_merging = 12.5;
        doorway_points_.clear();
        std::cout << "You have chosen the Voronoi random field segmentation." << std::endl;
        continue;
      }

      if (room_segmentation_algorithm == 6) { //xgboost
        doorway_points_.clear();
        double room_lower_limit_semantic_ = 1.0;
        double room_upper_limit_semantic_ = 1000000.;
        std::cout << "You have chosen the xgboost segmentation." << std::endl;
        XgboostClassifier xgClassifier(xgboost_path);
        xgClassifier.segmentMap(original_img, features_to_classify, segmented_map, map_resolution, room_lower_limit_semantic_, room_upper_limit_semantic_);
      }

      if (room_segmentation_algorithm == 99) {
        // pass through segmentation: takes a map which is already separated into unconnected areas and returns these as the resulting segmentation in the format of this program
        // todo: closing operation explicitly for bad maps --> needs parameterization
        // original_img.convertTo(segmented_map, CV_32SC1, 256, 0);
        // occupied space = 0, free space = 65280
        cv::Mat original_img_eroded, temp;
        cv::erode(original_img, temp, cv::Mat(), cv::Point(-1, -1), 3);
        cv::dilate(temp, original_img_eroded, cv::Mat(), cv::Point(-1, -1), 3);
        original_img_eroded.convertTo(segmented_map, CV_32SC1, 256, 0);     // occupied space = 0, free space = 65280
        int label_index = 1;
        double room_upper_limit_passthrough_ = 1000000.0;
        double room_lower_limit_passthrough_ = 1.0;

        //cv::imshow("original_img", original_img_eroded);
        //cv::waitKey();

        for (int y = 0; y < segmented_map.rows; y++) {
          for (int x = 0; x < segmented_map.cols; x++) {
            // if original map is occupied space here or if the segmented map has already received a label for that cell --> skip
            if (original_img_eroded.at<uchar>(y, x) != 255 || segmented_map.at<int>(y, x) != 65280)
            {
              continue;
            }

            // fill each room area with a unique id
            cv::Rect rect;
            cv::floodFill(segmented_map, cv::Point(x, y), label_index, &rect, 0, 0, 4);

            // determine filled area
            double area = 0;

            for (int v = rect.y; v < segmented_map.rows; v++)
              for (int u = rect.x; u < segmented_map.cols; u++)
                if (segmented_map.at<int>(v, u) == label_index)
                {
                  area += 1.;
                }

            area = map_resolution * map_resolution * area;  // convert from cells to m^2

            // exclude too small and too big rooms
            if (area < room_lower_limit_passthrough_ || area > room_upper_limit_passthrough_) {
              for (int v = rect.y; v < segmented_map.rows; v++)
                for (int u = rect.x; u < segmented_map.cols; u++)
                  if (segmented_map.at<int>(v, u) == label_index)
                  {
                    segmented_map.at<int>(v, u) = 0;
                  }
            }
            else
            {
              label_index++;
            }
          }
        }
      }

      cv::Mat index_map;
      PostProcessor post;
      ipa_building_msgs::MapSegmentationResult result;
      post.Process(segmented_map, doorway_points_, map_origin, index_map, result, 0.3, map_resolution);

      //do not need original segmented map below.
      segmented_map = index_map.clone();
      std::cout << "Depth of Segmented Map: " << segmented_map.depth() << ", Number of Channels: " << segmented_map.channels() << std::endl;

      // generate colored segmented_map
      cv::Mat color_segmented_map;
      segmented_map.convertTo(color_segmented_map, CV_8U);
      cv::cvtColor(color_segmented_map, color_segmented_map, cv::COLOR_GRAY2BGR);

      for (size_t i = 1; i <= result.room_information_in_pixel.size(); ++i) {
        //choose random color for each room
        const cv::Vec3b color((rand() % 250) + 1, (rand() % 250) + 1, (rand() % 250) + 1);

        for (size_t v = 0; v < segmented_map.rows; ++v)
          for (size_t u = 0; u < segmented_map.cols; ++u)
            if (segmented_map.at<int>(v, u) == i)
              color_segmented_map.at<cv::Vec3b>(v, u) = color;
      }

      std::string image_filename = segmented_map_path + map_name + "_segmented_" + segmentation_names[segmentation_index] + ".png";
      cv::imwrite(image_filename, color_segmented_map);
      std::cout << "segmented map saved to " << image_filename << std::endl;

      // evaluation: numeric properties
      // ==============================
      std::cout << "evaluation: numeric properties " << std::endl;
      std::vector<double> areas;
      std::vector<double> perimeters;
      std::vector<double> area_perimeter_compactness;
      std::vector<double> bb_area_compactness;
      std::vector<double> pca_eigenvalue_ratio;

      calculate_basic_measures(segmented_map, map_resolution, static_cast<int>(result.room_information_in_pixel.size()), areas, perimeters, area_perimeter_compactness, bb_area_compactness, pca_eigenvalue_ratio);

      // runtime
      results[segmentation_index].at<double>(0, image_index) = 0.0;

      // number of segments
      results[segmentation_index].at<double>(1, image_index) = segments_number_vector[segmentation_index] = areas.size();

      // area
      //std::vector<double> areas = calculate_areas_from_segmented_map(segmented_map, (int)result->room_information_in_pixel.size());
      double average = 0.0;
      double max_area = 0.0;
      double min_area = 100000000;
      calculate_mean_min_max(areas, average, min_area, max_area);
      results[segmentation_index].at<double>(2, image_index) = av_area_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(3, image_index) = max_area_vector[segmentation_index] = max_area;
      results[segmentation_index].at<double>(4, image_index) = min_area_vector[segmentation_index] = min_area;
      results[segmentation_index].at<double>(5, image_index) = dev_area_vector[segmentation_index] = calculate_stddev(areas, average);

      // perimeters
      //std::vector<double> perimeters = calculate_perimeters(saved_contours);
      average = 0.0;
      double max_per = 0.0;
      double min_per = 100000000;
      calculate_mean_min_max(perimeters, average, min_per, max_per);
      results[segmentation_index].at<double>(6, image_index) = av_per_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(7, image_index) = max_per_vector[segmentation_index] = max_per;
      results[segmentation_index].at<double>(8, image_index) = min_per_vector[segmentation_index] = min_per;
      results[segmentation_index].at<double>(9, image_index) = dev_per_vector[segmentation_index] = calculate_stddev(perimeters, average);

      // area compactness
      //std::vector<double> area_perimeter_compactness = calculate_compactness(saved_contours);
      average = 0.0;
      double max_compactness = 0;
      double min_compactness = 100000000;
      calculate_mean_min_max(area_perimeter_compactness, average, min_compactness, max_compactness);
      results[segmentation_index].at<double>(10, image_index) = av_compactness_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(11, image_index) = max_compactness_vector[segmentation_index] = max_compactness;
      results[segmentation_index].at<double>(12, image_index) = min_compactness_vector[segmentation_index] = min_compactness;
      results[segmentation_index].at<double>(13, image_index) = dev_compactness_vector[segmentation_index] = calculate_stddev(area_perimeter_compactness, average);

      // bounding box
      //std::vector<double> bb_area_compactness = calculate_bounding_error(saved_contours);
      average = 0.0;
      double max_error = 0;
      double min_error = 10000000;
      calculate_mean_min_max(bb_area_compactness, average, min_error, max_error);
      results[segmentation_index].at<double>(14, image_index) = av_bb_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(15, image_index) = max_bb_vector[segmentation_index] = max_error;
      results[segmentation_index].at<double>(16, image_index) = min_bb_vector[segmentation_index] = min_error;
      results[segmentation_index].at<double>(17, image_index) = dev_bb_vector[segmentation_index] = calculate_stddev(bb_area_compactness, average);

      // quotient
      //std::vector<double> pca_eigenvalue_ratio = calc_Ellipse_axis(saved_contours);
      average = 0.0;
      double max_quo = 0.0;
      double min_quo = 100000000;
      calculate_mean_min_max(pca_eigenvalue_ratio, average, min_quo, max_quo);
      results[segmentation_index].at<double>(18, image_index) = av_quo_vector[segmentation_index] = average;
      results[segmentation_index].at<double>(19, image_index) = max_quo_vector[segmentation_index] = max_quo;
      results[segmentation_index].at<double>(20, image_index) = min_quo_vector[segmentation_index] = min_quo;
      results[segmentation_index].at<double>(21, image_index) = dev_quo_vector[segmentation_index] = calculate_stddev(pca_eigenvalue_ratio, average);

      //          // retrieve room contours
      //          cv::Mat temporary_map = segmented_map.clone();
      //          std::vector<std::vector<cv::Point> > contours, saved_contours;
      //          std::vector<cv::Vec4i> hierarchy;
      //          for(size_t i = 1; i <= result->room_information_in_pixel.size(); ++i)
      //          {
      //              cv::Mat single_room_map = cv::Mat::zeros(segmented_map.rows, segmented_map.cols, CV_8UC1);
      //              for(size_t v = 0; v < segmented_map.rows; ++v)
      //                  for(size_t u = 0; u < segmented_map.cols; ++u)
      //                      if(segmented_map.at<int>(v,u) == i)
      //                          single_room_map.at<uchar>(v,u) = 255;
      //              cv::findContours(single_room_map, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
      //              cv::drawContours(temporary_map, contours, -1, cv::Scalar(0), cv::FILLED);
      //              for (int c = 0; c < contours.size(); c++)
      //              {
      //                  if (map_resolution * map_resolution * cv::contourArea(contours[c]) > 1.0)
      //                  {
      //                      saved_contours.push_back(contours[c]);
      //                  }
      //              }
      //          }
      //          // reachability
      //          if (check_reachability(saved_contours, segmented_map))
      //          {
      //              reachable[segmentation_index] = true;
      //          }
      //          else
      //          {
      //              reachable[segmentation_index] = false;
      //          }

      std::cout << "Basic measures computed." << std::endl;

      // evaluation: against ground truth segmentation
      // =============================================
      // load ground truth segmentation (just borders painted in between of rooms/areas, not colored yet --> coloring will be done here)
      bool has_gt = false;
      if (has_gt) {
        std::string map_name_basic = map_name;
        std::size_t pos = map_name.find("_furnitures");

        if (pos != std::string::npos)
          map_name_basic = map_name.substr(0, pos);

        std::string gt_image_filename = package_path + map_name_basic + "_gt_segmentation.png";
        std::cout << "Loading ground truth segmentation from: " << gt_image_filename << std::endl;
        cv::Mat gt_map = cv::imread(gt_image_filename.c_str(), CV_8U);

        // compute recall and precision, store colored gt segmentation
        double precision_micro, precision_macro, recall_micro, recall_macro;
        cv::Mat gt_map_color;
        EvaluationSegmentation es;
        es.computePrecisionRecall(gt_map, gt_map_color, segmented_map, precision_micro, precision_macro, recall_micro, recall_macro, true);
        std::string gt_image_filename_color = segmented_map_path + map_name + "_gt_color_segmentation.png"; //ros::package::getPath("ipa_room_segmentation") + "/common/files/test_maps/" + map_name + "_gt_color_segmentation.png";
        cv::imwrite(gt_image_filename_color.c_str(), gt_map_color);

        results[segmentation_index].at<double>(22, image_index) = recall_micro;
        results[segmentation_index].at<double>(23, image_index) = recall_macro;
        results[segmentation_index].at<double>(24, image_index) = precision_micro;
        results[segmentation_index].at<double>(25, image_index) = precision_macro;
      }
    }

    //write parameters into file
    std::stringstream output;
    output << "--------------Segmentierungsevaluierung----------------" << std::endl;

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << segmentation_names[i] << " & ";
    }

    output << std::endl;
    output << "Kompaktheitsmaße: " << std::endl;
    output << "Durschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_compactness_vector[i] << " & ";
    }

    output << std::endl;
    output << "****************************" << std::endl;

    output << "Überflüssige Fläche Bounding Box: " << std::endl;
    output << "Durchschnitt Bounding Fehler: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_bb_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Flächenmaße: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_area_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Umfangsmaße: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_per_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    //      output << "Erreichbarkeit für alle Raumzentren: " << std::endl;
    //      for(size_t i = 0; i < segmentation_names.size(); ++i)
    //      {
    //          if(reachable[i] == true)
    //              output << "Alle Raumzentren erreichbar" << std::endl;
    //          else
    //              output << "Nicht alle erreichbar" << std::endl;
    //      }
    //      output << "****************************" << std::endl;

    output << "Quotienten der Ellipsenachsen: " << std::endl;
    output << "Durchschnitt: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << av_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Maximum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << max_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Minimum: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << min_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "Standardabweichung: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << dev_quo_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Anzahl Räume: ";

    for (size_t i = 0; i < segmentation_names.size(); ++i)
    {
      output << segments_number_vector[i] << " & ";
    }

    output << std::endl;
    output << "**************************************" << std::endl;

    output << "Recall/Precision: " << std::endl;
    output << "recall_micro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(22, image_index) << " & ";
    }

    output << std::endl;
    output << "recall_macro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(23, image_index) << " & ";
    }

    output << std::endl;
    output << "precision_micro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(24, image_index) << " & ";
    }

    output << std::endl;
    output << "precision_macro: ";

    for (size_t i = 0; i < results.size(); ++i)
    {
      output << results[i].at<double>(25, image_index) << " & ";
    }

    output << std::endl;

    std::string log_filename = segmented_map_path + map_name + "_evaluation.txt";
    std::ofstream file(log_filename.c_str(), std::ios::out);

    if (file.is_open() == true)
    {
      file << output.str();
    }

    file.close();

    // write results summary to file (overwrite in each cycle in order to avoid loosing all data on crash)
    for (size_t segmentation_index = 0; segmentation_index < segmentation_names.size(); ++segmentation_index) {
      std::string log_filename = segmented_map_path + segmentation_names[segmentation_index] + "_evaluation_summary.txt";
      std::ofstream file(log_filename.c_str(), std::ios::out);

      if (file.is_open() == true) {
        for (int r = 0; r < results[segmentation_index].rows; ++r) {
          for (int c = 0; c < results[segmentation_index].cols; ++c)
          {
            file << results[segmentation_index].at<double>(r, c) << "\t";
          }

          file << std::endl;
        }
      }

      file.close();
    }
  }

  return 0;
}
