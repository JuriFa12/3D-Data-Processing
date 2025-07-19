#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "sgm.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#define NUM_DIRS 3
#define PATHS_PER_SCAN 8

using namespace std;
using namespace cv;
using namespace Eigen;
static char hamLut[256][256];
static int directions[NUM_DIRS] = {0, -1, 1};

//compute values for hamming lookup table
void compute_hamming_lut()
{
  for (uchar i = 0; i < 255; i++)
  {
    for (uchar j = 0; j < 255; j++)
    {
      uchar census_xor = i^j;
      uchar dist=0;
      while(census_xor)
      {
        ++dist;
        census_xor &= census_xor-1;
      }
      
      hamLut[i][j] = dist;
    }
  }
}

namespace sgm 
{
  SGM::SGM(unsigned int disparity_range, unsigned int p1, unsigned int p2, float conf_thresh, unsigned int window_height, unsigned window_width):
  disparity_range_(disparity_range), p1_(p1), p2_(p2), conf_thresh_(conf_thresh), window_height_(window_height), window_width_(window_width)
  {
    compute_hamming_lut();
  }

  // set images and initialize all the desired values
  void SGM::set(const  cv::Mat &left_img, const  cv::Mat &right_img, const  cv::Mat &right_mono)
  {
    views_[0] = left_img;
    views_[1] = right_img;
    mono_ = right_mono;


    height_ = left_img.rows;
    width_ = right_img.cols;
    pw_.north = window_height_/2;
    pw_.south = height_ - window_height_/2;
    pw_.west = window_width_/2;
    pw_.east = width_ - window_height_/2;
    init_paths();
    cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    inv_confidence_.resize(height_, vector<float>(width_));
    aggr_cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    path_cost_.resize(PATHS_PER_SCAN, ul_array3D(height_, ul_array2D(width_, ul_array(disparity_range_)))
    );
    
  }

  //initialize path directions
  void SGM::init_paths()
  {
    for(int i = 0; i < NUM_DIRS; ++i)
    {
      for(int j = 0; j < NUM_DIRS; ++j)
      {
        // skip degenerate path
        if (i==0 && j==0)
          continue;
        paths_.push_back({directions[i], directions[j]});
      }
    }
  }

  //compute costs and fill volume cost cost_
  void SGM::calculate_cost_hamming()
  {
    uchar census_left, census_right, shift_count;
    cv::Mat_<uchar> census_img[2];
    cv::Mat_<uchar> census_mono[2];
    cout << "\nApplying Census Transform" <<endl;
    
    for( int view = 0; view < 2; view++)
    {
      census_img[view] = cv::Mat_<uchar>::zeros(height_,width_);
      census_mono[view] = cv::Mat_<uchar>::zeros(height_,width_);

      for (int r = 1; r < height_ - 1; r++)
      {
        uchar *p_center = views_[view].ptr<uchar>(r),
              *p_census = census_img[view].ptr<uchar>(r);
        p_center += 1;
        p_census += 1;

        for(int c = 1; c < width_ - 1; c++, p_center++, p_census++)
        {
          uchar p_census_val = 0, m_census_val = 0, shift_count = 0;
          for (int wr = r - 1; wr <= r + 1; wr++)
          {
            for (int wc = c - 1; wc <= c + 1; wc++)
            {

              if( shift_count != 4 )//skip the center pixel
              {
                p_census_val <<= 1;
                m_census_val <<= 1;
                if(views_[view].at<uchar>(wr,wc) < *p_center ) //compare pixel values in the neighborhood
                  p_census_val = p_census_val | 0x1;

              }
              shift_count ++;
            }
          }
          *p_census = p_census_val;
        }
      }
    }

    cout <<"\nFinding Hamming Distance" <<endl;
    
    for(int r = window_height_/2 + 1; r < height_ - window_height_/2 - 1; r++)
    {
      for(int c = window_width_/2 + 1; c < width_ - window_width_/2 - 1; c++)
      {
        for(int d=0; d<disparity_range_; d++)
        {
          long cost = 0;
          for(int wr = r - window_height_/2; wr <= r + window_height_/2; wr++)
          {
            uchar *p_left = census_img[0].ptr<uchar>(wr),
                  *p_right = census_img[1].ptr<uchar>(wr);


            int wc = c - window_width_/2;
            p_left += wc;
            p_right += wc + d;



            const uchar out_val = census_img[1].at<uchar>(wr, width_ - window_width_/2 - 1);


            for(; wc <= c + window_width_/2; wc++, p_left++, p_right++)
            {
              uchar census_left, census_right, m_census_left, m_census_right;
              census_left = *p_left;
              if (c+d < width_ - window_width_/2)
              {
                census_right= *p_right;

              }

              else
              {
                census_right= out_val;
              }


              cost += ((hamLut[census_left][census_right]));
            }
          }
          cost_[r][c][d]=cost;
        }
      }
    }
  }

  void SGM::compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path)
  {
    unsigned long prev_cost, best_prev_cost, no_penalty_cost, penalty_cost, 
                  small_penalty_cost, big_penalty_cost;

    // if the processed pixel is the first:
    if(cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
    {
      for (int d = 0; d < disparity_range_; d++)
      {
        path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d]; //for all different values of d, set the path cost with the value of the cost volume
      }
   
    }

    else
    {

      best_prev_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][0];

      //Loop used to get the best cost of the previous pixel in the path direction
      for (int d = 1; d < disparity_range_; d++)
        {
          if (path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d] < best_prev_cost)
          {
            best_prev_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d];
          }
        }
          

      for (int d = 0; d < disparity_range_; d++)
      {
        /*For all different values of d we calculate the different pieces of the equation needed
        to finally calculate the path cost of the current pixel*/

        prev_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d];
        small_penalty_cost = min({path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d-1] + p1_, path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d+1] + p1_});
        big_penalty_cost = best_prev_cost + p2_;
        penalty_cost = min ({prev_cost, small_penalty_cost, big_penalty_cost});
        no_penalty_cost = cost_[cur_y][cur_x][d];
        path_cost_[cur_path][cur_y][cur_x][d] = no_penalty_cost + penalty_cost - best_prev_cost;

      }
 
    }
    
    
  }

  
  void SGM::aggregation()
  {
    //for all defined paths
    for(int cur_path = 0; cur_path < PATHS_PER_SCAN; ++cur_path)
    {


      int dir_x = paths_[cur_path].direction_x;
      int dir_y = paths_[cur_path].direction_y;
      
      int start_x, start_y, end_x, end_y, step_x, step_y;

      /*Through a cascade of ifs examinate the path direction and set the starting point, the ending point
      and the steps values in order to correctly iterate the image*/

      if (dir_x == 1 && dir_y == 1)
      {
        start_x = pw_.west ;
        start_y = pw_.north ;
        end_x = pw_.east;
        end_y = pw_.south;
        step_x = dir_x;
        step_y = dir_y;
      }
      else
      {
        if (dir_x == -1 && dir_y == -1)
        {
          start_x = pw_.east;
          start_y = pw_.south;
          end_x = pw_.west;
          end_y = pw_.north;
          step_x = dir_x;
          step_y = dir_y;
        }
        else
        {
          if (dir_x == 1 && dir_y == 0)
          {
            start_x = pw_.west;
            start_y = pw_.north;
            end_x = pw_.east;
            end_y = pw_.south;
            step_x = dir_x;
            step_y = 1;
          }
          else
          {
            if (dir_x == -1 && dir_y == 0)
            {
              start_x = pw_.east;
              start_y = pw_.north;
              end_x = pw_.west;
              end_y = pw_.south;
              step_x = dir_x;
              step_y = 1;
            }
            else
            {
              if (dir_x == 0 && dir_y == 1)
              {
              start_x = pw_.west;
              start_y = pw_.north;
              end_x = pw_.east;
              end_y = pw_.south;
              step_x = 1;
              step_y = 1;
              }
              else
              {
                if (dir_x == 0 && dir_y == -1)
                {
                  start_x = pw_.east;
                  start_y = pw_.south;
                  end_x = pw_.west;
                  end_y = pw_.north;
                  step_x = -1;
                  step_y = -1;
                }
                else
                {
                  if (dir_x == 1 && dir_y == -1)
                  {
                    start_x = pw_.west;
                    start_y = pw_.south;
                    end_x = pw_.east;
                    end_y = pw_.north;
                    step_x = +1;
                    step_y = -1;
                  }
                  else
                  {
                    if (dir_x == -1 && dir_y == 1)
                    {
                      start_x = pw_.east;
                      start_y = pw_.north;
                      end_x = pw_.west;
                      end_y = pw_.south;
                      step_x = -1;
                      step_y = +1;
                    }
                  }
                }
              }
            }
          }
        }
      }

      
      for(int y = start_y; y != end_y ; y+=step_y)
      {
        for(int x = start_x; x != end_x ; x+=step_x)
        {
          compute_path_cost(dir_y, dir_x, y, x, cur_path);
        }
      }
      
    }
    
    float alpha = (PATHS_PER_SCAN - 1) / static_cast<float>(PATHS_PER_SCAN);
    //aggregate the costs
    for (int row = 0; row < height_; ++row)
    {
      for (int col = 0; col < width_; ++col)
      {
        for(int path = 0; path < PATHS_PER_SCAN; path++)
        {
          unsigned long min_on_path = path_cost_[path][row][col][0];
          int disp =  0;

          for(int d = 0; d<disparity_range_; d++)
          {
            aggr_cost_[row][col][d] += path_cost_[path][row][col][d];
            if (path_cost_[path][row][col][d]<min_on_path)
              {
                min_on_path = path_cost_[path][row][col][d];
                disp = d;
              }

          }
          inv_confidence_[row][col] += (min_on_path - alpha * cost_[row][col][disp]);

        }
      }
    }

  }


  void SGM::compute_disparity()
  {

      vector<double> dmono_values;
      vector<double> dsgm_values;

      calculate_cost_hamming();
      aggregation();
      disp_ = Mat(Size(width_, height_), CV_8UC1, Scalar::all(0));
      int n_valid = 0;
      for (int row = 0; row < height_; ++row)
      {
          for (int col = 0; col < width_; ++col)
          {
              unsigned long smallest_cost = aggr_cost_[row][col][0];
              int smallest_disparity = 0;
              for(int d=disparity_range_-1; d>=0; --d)
              {

                  if(aggr_cost_[row][col][d]<smallest_cost)
                  {
                      smallest_cost = aggr_cost_[row][col][d];
                      smallest_disparity = d; 

                  }
              }
              inv_confidence_[row][col] = smallest_cost - inv_confidence_[row][col];

              // If the following condition is true, the disparity at position (row, col) has a good confidence
              if (inv_confidence_[row][col] > 0 && inv_confidence_[row][col] <conf_thresh_)
              {


                /*Store the disparity values obtained through Mono and SGM algorithm*/

                dmono_values.push_back((mono_.at<uchar>(row, col)));
                dsgm_values.push_back(smallest_disparity*255.0/disparity_range_);
                             
                /////////////////////////////////////////////////////////////////////////////////////////
              }

              disp_.at<uchar>(row, col) = smallest_disparity*255.0/disparity_range_;

          }
      }


      int N = dsgm_values.size();
  
      Matrix<double, Dynamic, 1> d_sgm(N, 1); //Create a column vector in order to store the sgm values
      d_sgm.col(0) = Map<VectorXd>(dsgm_values.data(), dsgm_values.size()); //Put the values in the first column
      

      Matrix<double, Dynamic, 2> d_mono(N, 2); //Create the matrix for the mono values
      

      d_mono.col(0) = Map<VectorXd>(dmono_values.data(), dmono_values.size()); //Put the values in the firs column
      d_mono.col(1) = VectorXd::Ones(dmono_values.size()); //Put all 1s in the second one

      //Series of operations used to calculate h and k

      MatrixXd AtA = d_mono.transpose() * d_mono;
      VectorXd Atb = d_mono.transpose() * d_sgm;
      VectorXd x = AtA.inverse() * Atb;
      

      /*We iterate all the pixels in the image and the one with low confidence values are substituted 
      with the scaled mono ones*/
      for (int row = 0; row < height_; ++row) 
      {
        for (int col = 0; col < width_; ++col) 
        {
            uchar old_value = mono_.at<uchar>(row, col);
            mono_.at<uchar>(row, col) = x(0) * old_value + x(1);

            if ((inv_confidence_[row][col] <= 0 || inv_confidence_[row][col]  >= conf_thresh_))
            {
              disp_.at<uchar>(row, col) = mono_.at<uchar>(row, col);
            }
              

        }
    }
        

    
  }

  float SGM::compute_mse(const cv::Mat &gt)
  {
    cv::Mat1f container[2];
    cv::normalize(gt, container[0], 0, 85, cv::NORM_MINMAX);
    cv::normalize(disp_, container[1], 0, disparity_range_, cv::NORM_MINMAX);

    cv::Mat1f  mask = min(gt, 1);
    cv::multiply(container[1], mask, container[1], 1);
    float error = 0;
    for (int y=0; y<height_; ++y)
    {
      for (int x=0; x<width_; ++x)
      {
        float diff = container[0](y,x) - container[1](y,x);
        error+=(diff*diff);
      }
    }
    error = error/(width_*height_);
    return error;
  }

  void SGM::save_disparity(char* out_file_name)
  {
    imwrite(out_file_name, disp_);
    return;
  }
  

}

