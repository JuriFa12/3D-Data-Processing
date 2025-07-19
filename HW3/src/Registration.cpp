#include "Registration.h"
#include <functional>

Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  source_for_icp_ = source_;
}

Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}

void Registration::draw_registration_result()
{
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  Eigen::Vector3d color_s(1, 0.706, 0);
  Eigen::Vector3d color_t(0, 0.651, 0.929);

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
}


void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // For each iteration:
    // 1. Find the closest point correspondences using find_closest_point().
    // 2. Use get_svd_icp_transformation() to estimate transformation.
    // 3. Apply transformation to source.
    // 4. Accumulate transformation and check RMSE convergence.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //Variable used to store concatenation of transformation
  //source_for_icp_ = source_.Transform(transformation_);   //Initialize source_for_icp
  source_for_icp_ = source_;
  source_for_icp_.Transform(transformation_); //Apply first transformation retrieved through descriptors

  //Initialzie RMSE to check convergence
  double prev_RMSE = std::numeric_limits<double>::infinity();
  double current_RMSE = 0.0;


  for (int i = 0; i < max_iteration; i++)
  {
    std::tuple< std::vector< size_t >, std::vector< size_t >, double > correspondences = find_closest_point( threshold );
    Eigen::Matrix4d current_transformation = get_svd_icp_transformation( std::get<0>(correspondences), std::get<1>(correspondences));
    current_RMSE = std::get<2>(correspondences); //Store RMSE here

    //Apply transformation to current cloud points'
    source_for_icp_.Transform(current_transformation);

    //Accumulate transformation
    transformation_ = current_transformation * transformation_;

    //Check convergence of RMSE
    if (std::abs(prev_RMSE - current_RMSE) < relative_rmse)
    {
      break;  //RMSE did not get better
    }
    prev_RMSE = current_RMSE; //Update RMSE to use to confront in next iteration


  }

}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1. Use KDTreeFlann to search the closest target point for each source point.
    // 2. If distance < threshold, record the pair and update RMSE.
    // 3. Return source indices, target indices, and final RMSE.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    open3d::geometry::KDTreeFlann tree (target_); //Create KDTreeFlann with target_ cloud points
    size_t num_points = source_for_icp_.points_.size(); //Obtain number of points observed

    //Create vectors to store index of closest target point and its distance from the point retrieved by searchKNN
    std::vector<int> retrieved_neighbor_index(1); 
    std::vector<double> distances(1);

    //Create vectors to store points with distance < than the threshold given
    std::vector<size_t> final_source_indices;
    std::vector<size_t> final_target_indices;

    double RMSE = 0.0; //Here the RMSE is stored

    size_t count = 0;

    for ( size_t i = 0; i < num_points; i++ )   //For each point in the point cloud
    {

      Eigen::Vector3d point = source_for_icp_.points_[i]; //Store the i point in a Vector of 3 doubles
      tree.SearchKNN( point, 1, retrieved_neighbor_index, distances ); //Search for the neighbor of source point
      double dist = std::sqrt(distances[0]); //Store the distance of the retrieved point from the source point

      if (dist < threshold ) //check if distance from source point is < than threshold
      {
        count ++; //Update counter of points which have distances < than the threshold

        //Store the index and distance in the vectors that will be returned by the function
        //Also update the RMSE to be returned in the end
        final_source_indices.push_back(i);
        final_target_indices.push_back( static_cast< size_t > (retrieved_neighbor_index[0]));
        RMSE += distances[0];

      }
    }

    if (count != 0) //Check if we have found some points
    {
      RMSE = sqrt(RMSE / double(count));

    }

    
    //Create tuple required and return it
    std::tuple<std::vector<size_t>, std::vector<size_t>, double> result(final_source_indices, final_target_indices, RMSE);

    return result;

}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 1. Compute centroids of source and target points.
  // 2. Subtract centroids and construct matrix H.
  // 3. Use Eigen::JacobiSVD to compute rotation.
  // 4. Handle special reflection case if det(R) < 0.
  // 5. Compute translation t and build 4x4 matrix.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //Extract points of correspondences
  std::vector < Eigen::Vector3d > source_corr, target_corr; //Vectors in which correspondences are stored
  Eigen::Vector3d s_centroid = Eigen::Vector3d::Zero(); //Vector in which source centroid is stored
  Eigen::Vector3d t_centroid = Eigen::Vector3d::Zero(); //Vector in which target centroid is stored

  for (size_t i = 0; i < source_indices.size(); i++) //For all points of correspondence between source and target
  {
    source_corr.push_back( source_for_icp_.points_[source_indices[i]] ); //Save point of correspondences i of source
    target_corr.push_back( target_.points_[target_indices[i]]); //Same for target

    s_centroid = s_centroid + source_corr[i]; //Add in order to get centroid in the end
    t_centroid = t_centroid + target_corr[i];

  }

  s_centroid = s_centroid / double(source_indices.size()); //Divide by number of elements to get real centroid value
  t_centroid = t_centroid / double(source_indices.size());

  for (size_t i = 0; i < source_indices.size(); i++) //Iterate through all correspondences
  {
    source_corr[i] = source_corr[i] - s_centroid; //Update them subtracting the relative centroid
    target_corr[i] = target_corr[i] - t_centroid;
  }

  //Create matrix where multiplication between points will be stored
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

  for (int i = 0; i < source_indices.size(); i++)
  {
    W += target_corr[i] * source_corr[i].transpose();  //Construct matrix W multiplying values of target and source
  }

  //Calculate SVD of W and get matrices U and V
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  //Get Matrix R multiplying U and V
  Eigen::Matrix3d R = U * V.transpose();

  //Check determinant of R
  if ( R.determinant() < 0)
  {
    Eigen::Matrix3d diagonal_matrix = Eigen::Matrix3d::Zero(); //Create diagonal matrix
    diagonal_matrix.diagonal() << 1, 1, -1; //Set diagonal values to 1 1 and -1
    R = U * diagonal_matrix * V.transpose(); //Obtain final R if corrupted data
  }

  //Obtain t in closed form
  Eigen::Vector3d translation_vector; 
  translation_vector = t_centroid - R * s_centroid; 



  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4, 4);
  transformation.block<3,3>(0,0) = R; //Place Rotation matrix in transformation
  transformation.block<3,1>(0,3) = translation_vector.cast<double>(); //Place t in transformation

  return transformation;
}


void Registration::execute_descriptor_registration()
{
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implement a registration method based entirely on manually implemented feature descriptors.
// - Preprocess the point clouds (e.g., downsampling).
// - Detect keypoints in both source and target clouds.
// - Compute descriptors manually (histogram-based, geometric, etc.) without any built-in functions.
// - Match descriptors and estimate initial correspondences.
// - Use RANSAC or other robust method to reject outliers and estimate an initial rigid transformation.
// - Do NOT use any part of ICP here; this must be a pure descriptor-based initial alignment.
// - Store the estimated transformation matrix in `transformation_`.
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //Downsampling point cloud points
  double voxel_size = 0.007;
  std::shared_ptr < open3d::geometry::PointCloud > source_downsampled = source_.VoxelDownSample(voxel_size);

  //Calculate keypoints for the downsampled cloud point
  std::shared_ptr < open3d::geometry::PointCloud > source_keypoints = open3d::geometry::keypoint::ComputeISSKeypoints( *source_downsampled );

  //Create KDTreeFlann to search neighbors of keypoints
  open3d::geometry::KDTreeFlann source_tree( *source_keypoints ); 

  //Get number of keypoints
  int n_source = source_keypoints -> points_.size();


  //Calculate keypoints normal
  source_keypoints -> EstimateNormals ( open3d::geometry::KDTreeSearchParamHybrid( voxel_size*2, 30 ) );

  //Number of bins used to discretize the angular features
  int bins = 7;

  //Max number of point that SearchKNN will retrieve
  int neighbors = 50;

  //Create vector to store values retrieved by SearchKNN and change their capacity to store at worst case all points
  //of source_keypoints
  std::vector< int > source_neighbor_indices;
  std::vector< double > source_neighbor_distances;

  //Reserve enough space for elements retrieved by SearchKNN
  source_neighbor_indices.reserve( neighbors );
  source_neighbor_distances.reserve( neighbors );


  //Prepare vectors which will store counts of bins of each angular feature of PFH
  std::vector < double > alpha_bins(bins, 0.0);
  std::vector < double > phi_bins(bins, 0.0); 
  std::vector < double > theta_bins(bins, 0.0);

  //Prepare vector in which each element i is the ith keypoint's descriptor for source
  std::vector < std::vector < double > > source_keypoint_descriptors;

  //Same but for target
  std::vector < std::vector < double > > target_keypoint_descriptors;
  

  //Compute descriptors for source's keypoints
  for (int i = 0; i < n_source; i++) //Iterate through all points of source
  {

    //Set to zero for each keypoint its feature vectors
    std::fill(alpha_bins.begin(), alpha_bins.end(), 0);
    std::fill(phi_bins.begin(),   phi_bins.end(),   0);
    std::fill(theta_bins.begin(), theta_bins.end(), 0);

    //Clear elements of vectors
    source_neighbor_indices.clear();
    source_neighbor_distances.clear();


    //Take point where descriptor is being calculated
    Eigen::Vector3d s = source_keypoints -> points_[i];

    //Take its normal
    Eigen::Vector3d normal_s = source_keypoints -> normals_[i];


    //Search for k neighbors of the given point and store number of actually found neighbors
    size_t true_neighbors = source_tree.SearchKNN( s, neighbors, source_neighbor_indices, source_neighbor_distances );

    if (true_neighbors <= 1) //Check that we have found some points
    {
      continue;
    }

    for (int j = 1; j < true_neighbors; j++) //start from j = 1 since searchknn includes point itself
    {
    
      double dist = std::sqrt(source_neighbor_distances[j]); //Get distances of neighbor from s
      int idx_q = source_neighbor_indices[j]; //Get index of point q which is being used in this iteration

      if (idx_q < 0 || idx_q >= n_source)  //Check range of idx_q
      {
        continue;
      }

      
      Eigen::Vector3d normal_t = source_keypoints -> normals_[idx_q]; //Take normale of q
      Eigen::Vector3d t = source_keypoints -> points_[idx_q]; //Take point q itself

  
      Eigen::Vector3d u = normal_s; //Set u as the normal vector of source point
      Eigen::Vector3d v = u.cross((t-s)/dist).normalized() ; //Set v following the formula, c'era normalized
      Eigen::Vector3d w = u.cross(v); //Set w following the formula     

      //Now set alpha, phi and theta
      double alpha = v.dot(normal_t);
      double phi = (u.dot((t - s))) / dist; 
      double theta = std::atan2( w.dot(normal_t), u.dot(normal_t));


      //For each feature, compute its bin in their relative vectors
      //Each element is clamped in order not to access memory parts which are not supposed to be touched
      double alpha_c = alpha;
      if (alpha_c < -1.0) alpha_c = -1.0;
      else if (alpha_c >  1.0) alpha_c =  1.0;
      int bin_alpha = int( std::floor( (alpha_c + 1.0) / 2.0 * bins ) ); //sum +1 to alpha since its range is [-1, 1] and divide by its total range
      bin_alpha = std::min( bin_alpha, bins - 1 ); // necessary in case bin_alpha is exactly "bins"
      alpha_bins[bin_alpha] ++; //Update value of bin with index bin_alpha

      
      double phi_c = phi;
      if (phi_c < -1.0) phi_c = -1.0;
      else if (phi_c >  1.0) phi_c =  1.0;
      int bin_phi = int( std::floor( (phi_c + 1.0) / 2.0 * bins) ); //same applies to phi since has same range of alpha
      bin_phi = std::min( bin_phi, bins - 1 ); //Same as before
      //phi_bins[bin_phi] ++;
      

      double theta_c = theta;
      if (theta_c < -M_PI) theta_c = -M_PI;
      else if (theta_c >  M_PI) theta_c =  M_PI;
      int bin_theta = int(std::floor((theta_c + M_PI)/(2.0*M_PI) * bins)); //same as before but theta has range [-pi, pi]
      bin_theta = std::min( bin_theta, bins - 1 ); //Same as before
      theta_bins[bin_theta] ++;

    }

    //Normalize each feature histogram
    for (int k = 0; k < bins; k++) 
    {
      alpha_bins[k] /= double(true_neighbors-1);
      phi_bins  [k] /= double(true_neighbors-1);
      theta_bins[k] /= double(true_neighbors-1);
    }

    //Create a descriptor concatenating the 3 features vectors and then store it in the keypoint descriptor vector
    std::vector< double > descriptor;
    descriptor.reserve(3 * bins); 
    descriptor.insert(descriptor.end(), alpha_bins.begin(),   alpha_bins.end());
    descriptor.insert(descriptor.end(), phi_bins.begin(),     phi_bins.end());
    descriptor.insert(descriptor.end(), theta_bins.begin(),   theta_bins.end());

    source_keypoint_descriptors.push_back(descriptor);

  }
  
  //Same structures created as for source 
  std::shared_ptr < open3d::geometry::PointCloud > target_downsampled = target_.VoxelDownSample(voxel_size);

  std::shared_ptr < open3d::geometry::PointCloud > target_keypoints = open3d::geometry::keypoint::ComputeISSKeypoints( *target_downsampled);


  target_keypoints -> EstimateNormals ( open3d::geometry::KDTreeSearchParamHybrid( voxel_size*2, 30 ) );

  open3d::geometry::KDTreeFlann target_tree( *target_keypoints );
   
  int n_target = target_keypoints -> points_.size();

  //Create vector to store values retrieved by searchRadius and resize them
  std::vector< int > target_neighbor_indices;
  std::vector< double > target_neighbor_distances;

  target_neighbor_indices.reserve( neighbors );
  target_neighbor_distances.reserve( neighbors );

  

  //Now same loop but for target
  for (int i = 0; i < n_target; i++) //Iterate through all points of source
  {
   
    //Set to zero for each keypoint its feature vectors
    std::fill(alpha_bins.begin(), alpha_bins.end(), 0);
    std::fill(phi_bins.begin(),   phi_bins.end(),   0);
    std::fill(theta_bins.begin(), theta_bins.end(), 0);
  
    //Take point where descriptors is being calculated
    Eigen::Vector3d s = target_keypoints -> points_[i];
  
    //Take its normal
    Eigen::Vector3d normal_s = target_keypoints -> normals_[i];
  
    target_neighbor_indices.clear();
    target_neighbor_distances.clear();
  
    //Search for k neighbors of the given point and store number of actually found neighbors
    int true_neighbors = target_tree.SearchKNN( s, neighbors, target_neighbor_indices, target_neighbor_distances );

    if ( true_neighbors <= 1) //Check if we found some neighbor
      continue;
    
    for (int j = 1; j < true_neighbors; j++) //start from j = 1 since searchknn includes point itself
    {
      double dist = std::sqrt(target_neighbor_distances[j]); //Leva sqrt in caso
      int idx_q = target_neighbor_indices[j]; //Get index of point q which is being used in this iteration

      if (idx_q < 0 || idx_q >= n_target) 
      {
        continue;
      }

      Eigen::Vector3d normal_t = target_keypoints -> normals_[idx_q]; //Take normale of q
      Eigen::Vector3d t = target_keypoints -> points_[idx_q]; //Take point q itself

      Eigen::Vector3d u = normal_s; //Set u as the normal vector of source point
      Eigen::Vector3d v = u.cross((t-s)/dist).normalized() ; //Set v following the formula
      Eigen::Vector3d w = u.cross(v); //Set w following the formula
        
  
      //Now set alpha, phi and theta
      double alpha = v.dot(normal_t);
      double phi = (u.dot((t - s))) / dist; 
      double theta = std::atan2( w.dot(normal_t), u.dot(normal_t));
      

      double alpha_c = alpha;
      if (alpha_c < -1.0) alpha_c = -1.0;
      else if (alpha_c >  1.0) alpha_c =  1.0;
      //For each feature, compute its bin in their relative vectors
      int bin_alpha = int( std::floor( (alpha_c + 1.0) / 2.0 * bins ) ); //sum +1 to alpha since its range is [-1, 1] and divide by its total range
      bin_alpha = std::min( bin_alpha, bins - 1 ); // necessary in case bin_alpha is exactly "bins"
      alpha_bins[bin_alpha] ++; //Update value of bin with index bin_alpha
      
      double phi_c = phi;
      if (phi_c < -1.0) phi_c = -1.0;
      else if (phi_c >  1.0) phi_c =  1.0;
      int bin_phi = int( std::floor( (phi_c + 1.0) / 2.0 * bins) ); //same applies to phi since has same range of alpha
      bin_phi = std::min( bin_phi, bins - 1 ); //Same as before
      //phi_bins[bin_phi] ++;

      double theta_c = theta;
      if (theta_c < -1.0) theta_c = -1.0;
      else if (theta_c >  1.0) theta_c =  1.0;
      int bin_theta = int(std::floor((theta_c + M_PI)/(2.0*M_PI) * bins)); //same as before but theta has range [-pi, pi]
      bin_theta = std::min( bin_theta, bins - 1 ); //Same as before
      theta_bins[bin_theta] ++;
  
    }
  
    //Normalize each feature histogram
    for (int k = 0; k < bins; k++) 
    {
      alpha_bins[k] /= double(true_neighbors-1);
      phi_bins  [k] /= double(true_neighbors-1);
      theta_bins[k] /= double(true_neighbors-1);
    }
  
    //Create a descriptor concatenating the 3 features vectors and then store it in the keypoint descriptor vector
    std::vector< double > descriptor;
    descriptor.reserve(3 * bins);  
    descriptor.insert(descriptor.end(), alpha_bins.begin(),   alpha_bins.end());
    descriptor.insert(descriptor.end(), phi_bins.begin(),     phi_bins.end());
    descriptor.insert(descriptor.end(), theta_bins.begin(),   theta_bins.end());

    target_keypoint_descriptors.push_back(descriptor);
  
  
  }



  //Matching phase
  double max_distance = 0.2; //Set a maximum distance between descriptors

  //Here 2 vectors for basically the same task are created so that in future a manual implementation
  //of RANSAC method is implemented
  std::vector < int > matches; //Vector where we store matches indices: matches[i] = j indicates a match between source[i] and target[j]
  open3d::pipelines::registration::CorrespondenceSet correspondences; //Vector which stores index of a correspondece in a vector 


  matches.resize(source_keypoint_descriptors.size(), -1);


  for (int i = 0; i < source_keypoint_descriptors.size(); i++) //For each keypoint in source

  {
    int best = -1; //Here index of match will be stored, -1 used as flag value

    std::vector < double > source_descriptor = source_keypoint_descriptors[i]; //Get the descriptor i

    double best_distance = std::numeric_limits<double>::infinity(); //Set initial distance to infinity

    for (int j = 0; j < target_keypoint_descriptors.size(); j++) //Iterate through all keypoints of target
    {
      std::vector < double > target_descriptor = target_keypoint_descriptors[j]; //Take descriptor j
      double sum = 0; //Set sum of distance to zero as initial value

      for (int k = 0; k < 3 * bins; k++) //Iterate through all elements of each descriptor
      {
        double d = double( source_descriptor[k]) - double ( target_descriptor[k]); //Subtract each component
        sum += d * d; //add its square to sum
      }

      if ( sum < best_distance ) //Check if descriptor j is more similar to i than the previosu best one
      {
        //If yes, update distance and index associated
        best_distance = sum;
        best = j;
      }
    }

    if ( std::sqrt(best_distance) <= max_distance ) //check if it not too distant
    {
      matches[i] = best; //Keypoint i has match with best=j
      correspondences.emplace_back(i, best); //Same but with a different element
    }
    else
    {
      matches[i] = -1; //Set match with a false value
    }

  }

  //RANSAC phase

  auto Ransac_result = open3d::pipelines::registration::RegistrationRANSACBasedOnCorrespondence( 
    *source_keypoints,
    *target_keypoints,
    correspondences,
    1.5 * voxel_size
  );
  

  transformation_ = Ransac_result.transformation_;

}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_ = init_transformation;
}

Eigen::Matrix4d Registration::get_transformation()
{
  return transformation_;
}


double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points = source_clone.points_.size();
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse = 0.0;

  for (size_t i = 0; i < num_source_points; ++i) {
    Eigen::Vector3d source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i / (i + 1) + dist2[0] / (i + 1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile(filename);
  if (outfile.is_open()) {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;
  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone + source_clone;
  open3d::io::WritePointCloud(filename, merged);
}


