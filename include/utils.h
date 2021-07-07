#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>
#include <ctime>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <iterator>
#include <boost/filesystem.hpp>
#include <boost/spirit/include/qi.hpp>

#include <xgboost/logging.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>
#include <xgboost/c_api.h>

#include <ipa_room_segmentation/features.h>
#include <ipa_room_segmentation/raycasting.h>

cv::Vec3b white(255, 255, 255);

using namespace xgboost;

namespace qi = boost::spirit::qi;

using Vector32f = std::vector<float>;
using Matrix32f = std::vector<std::vector<float>>;

// http://stackoverflow.com/a/1764367
template <typename Iterator>
bool loadCSV(Iterator first, Iterator last, Matrix32f& x)
{
  return boost::spirit::qi::phrase_parse(first, last, +(qi::float_ % ','), qi::space, x);
}

static bool loadCSV(std::istream& is, Matrix32f& x)
{
  bool r = false;
  if (is)
  {
    // wrap istream into iterator
    is.unsetf(std::ios::skipws);
    boost::spirit::istream_iterator first(is), last;
    r = loadCSV(first, last, x);
  }
  return r;
}

static bool loadCSV(const std::string& filename, Matrix32f& x)
{
  std::ifstream is(filename);
  return loadCSV(is, x);
}

// http://stackoverflow.com/a/1764367
template <typename Iterator>
bool loadCSV(Iterator first, Iterator last, Matrix32f& x, Vector32f& y)
{
  bool r = boost::spirit::qi::phrase_parse(first, last, +(qi::float_ % ','), qi::space, x);
  if (r)
  {
    x.resize(y.size());
    for (int i = 0; i < x.size(); i++)
    {
      y[i] = x[i].back();
      x[i].pop_back();
    }
  }
  return r;
}

static bool loadCSV(std::istream& is, Matrix32f& v, Vector32f& x)
{
  bool r = false;
  if (is)
  {
    // wrap istream into iterator
    is.unsetf(std::ios::skipws);
    boost::spirit::istream_iterator first(is), last;
    r = loadCSV(first, last, v, x);
  }
  return r;
}

static bool loadCSV(const std::string& filename, Matrix32f& v, Vector32f& x)
{
  std::ifstream is(filename);
  return loadCSV(is, v, x);
}

int getFileNames(const std::string& dir, std::vector<std::string>& filenames)
{
  boost::filesystem::path path(dir);
  if (!boost::filesystem::exists(path))
  {
    return -1;
  }

  boost::filesystem::directory_iterator end_iter;
  for (boost::filesystem::directory_iterator iter(path); iter != end_iter; ++iter)
  {
    if (boost::filesystem::is_regular_file(iter->status()))
    {
      filenames.push_back(iter->path().string());
    }

    if (boost::filesystem::is_directory(iter->status()))
    {
      getFileNames(iter->path().string(), filenames);
    }
  }

  return filenames.size();
}

int segmentationNameToNumber(const std::string name) {
  if (name.compare("1morphological") == 0)
  {
    return 1;
  }
  else
    if (name.compare("2distance") == 0)
    {
      return 2;
    }
    else
      if (name.compare("3voronoi") == 0)
      {
        return 3;
      }
      else
        if (name.compare("4semantic") == 0)
        {
          return 4;
        }
        else
          if (name.compare("5vrf") == 0)
          {
            return 5;
          }
          else
            if (name.compare("6xgboost") == 0)
            {
              return 6;
            }

  return 1;
}

void calculate_mean_min_max(const std::vector<double>& values, double& mean, double& min_val, double& max_val) {
  mean = 0.0;
  max_val = 0.0;
  min_val = 1e10;

  for (size_t i = 0; i < values.size(); ++i) {
    mean += values[i];

    if (values[i] > max_val)
    {
      max_val = values[i];
    }

    if (values[i] < min_val)
    {
      min_val = values[i];
    }
  }

  mean = mean / (double)values.size();
}

double calculate_stddev(const std::vector<double>& values, const double mean) {
  //calculate the standard deviation
  double sigma = 0.;

  for (size_t i = 0; i < values.size(); ++i)
  {
    sigma += (values[i] - mean) * (values[i] - mean);
  }

  sigma = std::sqrt(sigma / (double)(values.size() - 1.));

  return sigma;
}

bool check_inner_pixel(const cv::Mat& map, const int u, const int v) {
  const int label = map.at<int>(v, u);

  for (int dv = -1; dv <= 1; ++dv) {
    for (int du = -1; du <= 1; ++du) {
      const int nu = u + du;
      const int nv = v + dv;

      if (nu >= 0 && nu < map.cols && nv >= 0 && nv < map.rows && !(nu == 0 && nv == 0)) {
        if (map.at<int>(nv, nu) != label)
        {
          return false;
        }
      }
    }
  }

  return true;
}

void calculate_basic_measures(const cv::Mat& map, const double map_resolution, const int number_rooms, std::vector<double>& areas, std::vector<double>& perimeters,
  std::vector<double>& area_perimeter_compactness, std::vector<double>& bb_area_compactness, std::vector<double>& pca_eigenvalue_ratio) {
  std::cout << "number of rooms: " << number_rooms << std::endl;
  areas.clear();
  areas.resize(number_rooms, 0.);
  perimeters.clear();
  perimeters.resize(number_rooms, 0.);
  area_perimeter_compactness.clear();
  area_perimeter_compactness.resize(number_rooms, 0.);
  bb_area_compactness.clear();
  bb_area_compactness.resize(number_rooms, 0.);
  pca_eigenvalue_ratio.clear();
  pca_eigenvalue_ratio.resize(number_rooms, 0.);

  std::vector< std::vector< cv::Point > > room_contours(number_rooms);
  std::vector< std::vector< cv::Point > > filled_rooms(number_rooms);

  //const double map_resolution = 0.0500; // m/cell
  for (size_t v = 0; v < map.rows; ++v) {
    for (size_t u = 0; u < map.cols; ++u) {
      if (map.at<int>(v, u) != 0) {
        const int insert_index = map.at<int>(v, u) - 1;

        if (insert_index >= number_rooms)
          continue;

        //std::cout << "put data to areas filled_rooms at " << map.at<int>(v,u) << " index " <<  insert_index << std::endl;
        areas[insert_index] += map_resolution * map_resolution;
        //std::cout << "put data to areas done." << std::endl;
        filled_rooms[insert_index].push_back(cv::Point(u, v));

        //std::cout << "put data to filled_rooms done." << std::endl;
        if (check_inner_pixel(map, u, v) == false) {
          //std::cout << "put data to room_contours." << std::endl;
          room_contours[insert_index].push_back(cv::Point(u, v));
        }
      }
    }
  }

  for (size_t r = 0; r < room_contours.size(); ++r) {
    // perimeters
    perimeters[r] = map_resolution * room_contours[r].size();
    // area_perimeter_compactness
    area_perimeter_compactness[r] = areas[r] / (perimeters[r] * perimeters[r]);
    // bb_area_compactness
    cv::RotatedRect rotated_bounding_box = cv::minAreaRect(room_contours[r]);
    double bounding_box_area = map_resolution * map_resolution * rotated_bounding_box.size.area();
    bb_area_compactness[r] = areas[r] / bounding_box_area;
    // pca_eigenvalue_ratio
    cv::Mat data(filled_rooms[r].size(), 2, CV_64FC1);

    for (size_t i = 0; i < filled_rooms[r].size(); ++i) {
      data.at<double>(i, 0) = filled_rooms[r][i].x;
      data.at<double>(i, 1) = filled_rooms[r][i].y;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    pca_eigenvalue_ratio[r] = pca.eigenvalues.at<double>(0) / pca.eigenvalues.at<double>(1);
  }
}

//calculate the compactness of the rooms. Compactness factor is given by area/perimeter
std::vector<double> calculate_compactness(std::vector<std::vector<cv::Point> > rooms, const double map_resolution) {
  double current_area, current_perimeter;
  //double map_resolution = 0.05000;
  std::vector<double> compactness_factors;

  //calculate the area and perimeter for each room using opencv
  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    current_area = map_resolution * map_resolution * cv::contourArea(rooms[current_room]);
    current_perimeter = map_resolution * cv::arcLength(rooms[current_room], true);
    compactness_factors.push_back(current_area / (current_perimeter * current_perimeter));
  }

  return compactness_factors;
}

//calculate too much area of the bounding box
std::vector<double> calculate_bounding_error(std::vector<std::vector<cv::Point> > rooms, const double map_resolution) {
  std::vector<double> space_errors;
  double bounding_box_area, room_area;
  //double map_resolution = 0.05000;
  cv::RotatedRect current_bounding_box;

  //calculate the rotated bounding box for each room and subtract the roomarea from it
  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    current_bounding_box = cv::minAreaRect(rooms[current_room]);
    bounding_box_area = map_resolution * map_resolution * current_bounding_box.size.area();
    room_area = map_resolution * map_resolution * cv::contourArea(rooms[current_room]);
    //put the difference in the error vector
    space_errors.push_back(bounding_box_area - room_area);
  }

  return space_errors;
}

//calculate area for every room
std::vector<double> calculate_areas(std::vector<std::vector<cv::Point> > rooms, const double map_resolution) {
  std::vector<double> calculated_areas;

  //double map_resolution = 0.0500;
  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    calculated_areas.push_back(map_resolution * map_resolution * cv::contourArea(rooms[current_room]));
  }

  return calculated_areas;
}

//calculate area for every room
std::vector<double> calculate_areas_from_segmented_map(const cv::Mat& map, const double map_resolution, const int number_rooms) {
  std::vector<double> calculated_areas(number_rooms, 0.);

  //const double map_resolution = 0.0500; // m/cell
  for (size_t v = 0; v < map.rows; ++v)
    for (size_t u = 0; u < map.cols; ++u)
      if (map.at<int>(v, u) != 0)
      {
        calculated_areas[map.at<int>(v, u) - 1] += map_resolution * map_resolution;
      }

  return calculated_areas;
}

//calculate perimeter for every room
std::vector<double> calculate_perimeters(std::vector<std::vector<cv::Point> > rooms) {
  std::vector<double> calculated_perimeters;
  double map_resoultion = 0.0500;

  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    calculated_perimeters.push_back(map_resoultion * cv::arcLength(rooms[current_room], true));
  }

  return calculated_perimeters;
}

//check if every roomcenter is reachable
bool check_reachability(const std::vector<std::vector<cv::Point> >& rooms, const cv::Mat& map) {
  bool reachable = true;
  cv::RotatedRect current_bounding_box;

  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    current_bounding_box = minAreaRect(rooms[current_room]);

    if (map.at<unsigned char>(current_bounding_box.center) == 0) {
      reachable = false;
    }
  }

  return reachable;
}

//Calculate the length of the major axis of the minimum bounding ellipse
double calc_major_axis(std::vector<cv::Point> room) {
  cv::Point2f points[4];
  std::vector<cv::Point2f> edge_points;
  double distance = 0;
  double map_resoultion = 0.05;
  //saving-variable for the Points of the ellipse
  cv::RotatedRect ellipse = cv::fitEllipse(cv::Mat(room));
  //get the edge-points of the ellipse
  ellipse.points(points);

  //saving the Points of the ellipse in a vector
  for (int i = 0; i < 4; i++) {
    edge_points.push_back(points[i]);
  }

  //calculate the distance between the Points and take the largest one
  for (int p = 0; p < edge_points.size(); p++) {
    for (int np = 0; np < edge_points.size(); np++) {
      if (std::sqrt(std::pow((edge_points[p].x - edge_points[np].x), 2) + std::pow((edge_points[p].y - edge_points[np].y), 2)) > distance) {
        distance = std::sqrt(std::pow((edge_points[p].x - edge_points[np].x), 2)
          + std::pow((edge_points[p].y - edge_points[np].y), 2));
      }
    }
  }

  return map_resoultion * distance;
}

//Calculate the length of the minor axis of the minimum bounding ellipse
double calc_minor_axis(std::vector<cv::Point> room) {
  cv::Point2f points[4];
  std::vector<cv::Point2f> edge_points;
  double distance = 10000000;
  double map_resoultion = 0.05;
  //saving-variable for the Points of the ellipse
  cv::RotatedRect ellipse = cv::fitEllipse(cv::Mat(room));
  //get the edge-points of the ellipse
  ellipse.points(points);

  //saving the Points of the ellipse in a vector
  for (int i = 0; i < 4; i++) {
    edge_points.push_back(points[i]);
  }

  //calculate the distance between the Points and take the largest one
  for (int p = 0; p < edge_points.size(); p++) {
    for (int np = 0; np < edge_points.size(); np++) {
      //np != p: make sure the distance is nor calculated to the Point itself
      if (std::sqrt(std::pow((edge_points[p].x - edge_points[np].x), 2)
        + std::pow((edge_points[p].y - edge_points[np].y), 2)) < distance && np != p) {
        distance = std::sqrt(std::pow((edge_points[p].x - edge_points[np].x), 2)
          + std::pow((edge_points[p].y - edge_points[np].y), 2));
      }
    }
  }

  return map_resoultion * distance;
}

//Calculate the Quotient of the langths of the major axis and the minor axis from the fitting ellipse for each room
std::vector<double> calc_Ellipse_axis(std::vector<std::vector<cv::Point> > rooms) {
  std::vector<double> quotients;

  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    quotients.push_back(calc_major_axis(rooms[current_room]) / calc_minor_axis(rooms[current_room]));
  }

  return quotients;
}

//Calculate the average distance between room-centers
double calc_average_distance(std::vector<std::vector<cv::Point> > rooms) {
  double mean = 0.0;
  double dx, dy;
  double map_resoultion = 0.05;
  std::vector<cv::Point2f> centers;
  cv::RotatedRect current_bounding_box;

  for (int current_room = 0; current_room < rooms.size(); current_room++) {
    current_bounding_box = minAreaRect(rooms[current_room]);
    centers.push_back(current_bounding_box.center);
  }

  //calculate the sum of distances
  for (int current_center = 0; current_center < centers.size() - 1; current_center++) {
    dx = centers[current_center].x - centers[current_center + 1].x;
    dy = centers[current_center].y - centers[current_center + 1].y;
    mean += std::sqrt(std::pow(dx, 2.0) + std::pow(dy, 2.0));
  }

  return map_resoultion * (mean / centers.size());
}

//Calculate standard deviation of room-areas
double calc_area_deviation(std::vector<std::vector<cv::Point> > rooms, const double map_resolution) {
  double sigma = 0.0;
  double mean = 0.0;
  std::vector<double> areas = calculate_areas(rooms, map_resolution);

  //calculate the average room-area
  for (int current_room = 0; current_room < areas.size(); current_room++) {
    mean += areas[current_room];
  }

  mean = mean / areas.size();

  //calculate the standard deviation
  for (int current_room = 0; current_room < areas.size(); current_room++) {
    sigma += std::pow(areas[current_room] - mean, 2.0);
  }

  sigma = sigma / (areas.size() - 1);
  return std::sqrt(sigma);
}

//Calculate standard deviation of room-perimeters
double calc_perimeter_deviation(std::vector<std::vector<cv::Point> > rooms) {
  double sigma = 0.0;
  double mean = 0.0;
  std::vector<double> perimeters = calculate_perimeters(rooms);

  //calculate the average room-area
  for (int current_room = 0; current_room < perimeters.size(); current_room++) {
    mean += perimeters[current_room];
  }

  mean = mean / perimeters.size();

  //calculate the standard deviation
  for (int current_room = 0; current_room < perimeters.size(); current_room++) {
    sigma += std::pow(perimeters[current_room] - mean, 2.0);
  }

  sigma = sigma / (perimeters.size() - 1);
  return std::sqrt(sigma);
}

//Calculate standard deviation of ellipsis-quotients
double calc_quotients_deviation(std::vector<std::vector<cv::Point> > rooms) {
  double sigma = 0.0;
  double mean = 0.0;
  std::vector<double> quotients = calc_Ellipse_axis(rooms);

  //calculate the average room-area
  for (int current_room = 0; current_room < quotients.size();
    current_room++) {
    mean += quotients[current_room];
  }

  mean = mean / quotients.size();

  //calculate the standard deviation
  for (int current_room = 0; current_room < quotients.size();
    current_room++) {
    sigma += std::pow(quotients[current_room] - mean, 2.0);
  }

  sigma = sigma / (quotients.size() - 1);
  return std::sqrt(sigma);
}

//Calculate standard deviation of bounding-box-errors
double calc_errors_deviation(std::vector<std::vector<cv::Point> > rooms, const double map_resolution) {
  double sigma = 0.0;
  double mean = 0.0;
  std::vector<double> errors = calculate_bounding_error(rooms, map_resolution);

  //calculate the average room-area
  for (int current_room = 0; current_room < errors.size(); current_room++) {
    mean += errors[current_room];
  }

  mean = mean / errors.size();

  //calculate the standard deviation
  for (int current_room = 0; current_room < errors.size(); current_room++) {
    sigma += std::pow(errors[current_room] - mean, 2.0);
  }

  sigma = sigma / (errors.size() - 1);
  return std::sqrt(sigma);
}
