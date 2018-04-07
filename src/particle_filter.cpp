/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random> // Need this for sampling from distributions
#include <iostream>


#include "particle_filter.h"

using namespace std;

#define EPS 0.00001 

double dist_2(double x1, double y1, double x2, double y2) {
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}



bool debug = false;

void print_particle(std::vector<Particle> &particles, int i) {
  std::cout << "id: " << particles[i].id << " x: " << particles[i].x
            << " y: " << particles[i].y << " theta: " << particles[i].theta
            << " weight: " << particles[i].weight << endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  num_particles = 50;

  std::default_random_engine gen;

  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  if (debug)
    std::cout << "INIT" << endl; // DEBUG
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1;
    particles.push_back(p);
    weights.push_back(1);
    if (debug && i < 10) {
      print_particle(particles, i);
    } // DEBUG
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add velocity and yaw rate measurements to each particle and add
  // random Gaussian noise to predict the car's position (pose). NOTE: When
  // adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  if (debug)
    std::cout << "PREDICT" << endl; // DEBUG
  for (int i = 0; i < num_particles; ++i) {

    double x;
    double y;
    double theta;

    if (yaw_rate) {
      x = particles[i].x + velocity / yaw_rate *
                               (sin(particles[i].theta + yaw_rate * delta_t) -
                                sin(particles[i].theta));
      y = particles[i].y + velocity / yaw_rate *
                               (cos(particles[i].theta) -
                                cos(particles[i].theta + yaw_rate * delta_t));
      theta = particles[i].theta + yaw_rate * delta_t;
    } else {
      x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      theta = particles[i].theta;
    }

    std::normal_distribution<double> N_x(x, std_pos[0]);
    std::normal_distribution<double> N_y(y, std_pos[1]);
    std::normal_distribution<double> N_theta(theta, std_pos[2]);

    particles[i].x = N_x(gen);
    particles[i].y = N_y(gen);
    particles[i].theta = N_theta(gen);
    if (debug)
      print_particle(particles, i); // DEBUG
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement (position of landmark from the map)
  // that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  if (debug)
    std::cout << "LANDMARK ASSOCATION" << endl; // DEBUG

  for (int i = 0; i < observations.size(); ++i) {
    double distance_min;
    if (debug)
      std::cout << "Observation: " << i << endl; // DEBUG
    distance_min = 1000.0;
    for (int j = 0; j < predicted.size(); ++j) {
      double pred_to_obs;
      pred_to_obs = dist_2(predicted[j].x, predicted[j].y, observations[i].x,
                         observations[i].y);
      // std::cout << "landmark: " << predicted[j].id
      //<< " distance: " << pred_to_obs << endl; // DEBUG
      if (pred_to_obs < distance_min) {
        distance_min = pred_to_obs;
        observations[i].id = predicted[j].id;
      }
    }
    if (debug)
      std::cout << "associate landmark: " << observations[i].id
                << " distance: " << distance_min << endl; // DEBUG
  }
}

int lm_index_from_id(std::vector<LandmarkObs> predicted_lm, int id) {
  for (int i = 0; i < predicted_lm.size(); ++i) {
    if (predicted_lm[i].id == id) {
      return i;
    }
  }
  return -1;
}

double multi_gauss(double x, double y, double lm_x, double lm_y, double std_x,
                   double std_y) {
  double a = pow(x - lm_x, 2.0) / (2.0 * pow(std_x, 2.0));
  double b = pow(y - lm_y, 2.0) / (2.0 * pow(std_y, 2.0));
  double p = exp(-(a + b)) / (2.0 * M_PI * std_x * std_y);
  if (debug)
    std::cout << " multi gauss:  " << p << endl; // DEBUG
  return p;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

  if (debug) {
    std::cout << "UPDATE WEIGHTS" << endl; // DEBUG
    std::cout << "Landmarks: " << map_landmarks.landmark_list.size() << endl;
  } // DEBUG

  for (int i = 0; i < num_particles; ++i) {
    std::vector<LandmarkObs> predicted_lm;
    std::vector<LandmarkObs> transformed_obs;

    if (debug && i < 10) {
      std::cout << "Particle " << i << endl;
    } // DEBUG

    // Create a list of landmarks within range of particle sensors
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double d;
      d = dist_2(particles[i].x, particles[i].y,
               map_landmarks.landmark_list[j].x_f,
               map_landmarks.landmark_list[j].y_f);
      if (d <= sensor_range) {
        LandmarkObs lm;
        lm.id = map_landmarks.landmark_list[j].id_i;
        lm.x = map_landmarks.landmark_list[j].x_f;
        lm.y = map_landmarks.landmark_list[j].y_f;
        predicted_lm.push_back(lm);
        if (debug) {
          std::cout << "Landmark " << j << " x: " << lm.x << " y: " << lm.y
                    << " distance " << d; // DEBUG
          std::cout << " within sensor range" << endl;
        } // DEBUG
      }
    }

    if (debug)
      std::cout << "Transform observations" << endl; // DEBUG

    // Transform observations from car coordinates to map coordinates
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs lm;
      if (debug) {
        std::cout << "Observation " << j; // DEBUG
        std::cout << " x: " << observations[j].x << " y: " << observations[j].y
                  << endl;
      } // DEBUG
      lm.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) -
             (sin(particles[i].theta) * observations[j].y);
      lm.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) +
             (cos(particles[i].theta) * observations[j].y);
      transformed_obs.push_back(lm);
      if (debug) {
        std::cout << "Transformed " << j;
        std::cout << " x: " << lm.x << " y: " << lm.y << endl;
      } // DEBUG
    }

    // Associate each landmark observation with nearest map landmark
    dataAssociation(predicted_lm, transformed_obs);

    if (debug)
      std::cout << "Update weights using multi gauss" << endl; // DEBUG

    // Update particle weights with Multivariate-Gaussian Probability Density
    // Combine the probabilities to arrive at final weights (Posterior
    // Probability)
    double final_weight;
    final_weight = 1.0;
    for (int j = 0; j < transformed_obs.size(); ++j) {
      double lm_x, lm_y;
      int lm_index = lm_index_from_id(predicted_lm, transformed_obs[j].id);
      lm_x = predicted_lm[lm_index].x;
      lm_y = predicted_lm[lm_index].y;
      if (debug) {
        std::cout << "Landmark x: " << lm_x;
        std::cout << " Landmark y: " << lm_y;
      } // DEBUG
      final_weight *= multi_gauss(transformed_obs[j].x, transformed_obs[j].y,
                                  lm_x, lm_y, std_landmark[0], std_landmark[1]);
    }
    particles[i].weight = final_weight;
    weights[i] = final_weight;
    if (debug)
      std::cout << "Final weight: " << final_weight << endl; // DEBUG
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Set of resampled particles

  std::vector<Particle> resampled_p;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> d(weights.begin(), weights.end());

  for (int n = 0; n < num_particles; ++n) {
    resampled_p.push_back(particles[d(gen)]);
  }

  particles = resampled_p;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}


/*
// @param gps_x 	GPS provided x position
// @param gps_y 	GPS provided y position
// @param theta		GPS provided yaw
void ParticleFilter::gaussian_init(double gps_x, double gps_y, double theta, double std_x, double std_y, double std_theta, int num_particles) {
	default_random_engine gen;
	// Standard deviations for x, y, and theta

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(gps_x, std_x);
	normal_distribution<double> dist_y(gps_y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	for (int i = 0; i < num_particles; ++i) {
		double sample_x, sample_y, sample_theta;
		
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);	 

		Particle part;
		part.id = i;
		part.x = sample_x;
		part.y = sample_y;
		part.theta = sample_theta;
		part.weight = 1.0;
		
		weights.push_back(part.weight);
		particles.push_back(part);				
		 
	}

}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 50;	

	if(!is_initialized){
		std::default_random_engine gen;
		// Standard deviations for x, y, and theta

		// This line creates a normal (Gaussian) distribution for x, y and theta
		std::normal_distribution<double> dist_x(x, std[0]);
		std::normal_distribution<double> dist_y(y, std[1]);
		std::normal_distribution<double> dist_theta(theta, std[2]);

		for (int i = 0; i < num_particles; ++i) {
			double sample_x, sample_y, sample_theta;
			
			sample_x = dist_x(gen);
			sample_y = dist_y(gen);
			sample_theta = dist_theta(gen);	 

			Particle part;
			part.id = i;
			part.x = sample_x;
			part.y = sample_y;
			part.theta = sample_theta;
			part.weight = 1;
			
			weights.push_back(1);
			particles.push_back(part);				
				
		}
	}
	
	bool is_initialized = true;
	
	cout << "Initialized!"<< endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	cout << "Prediction" << endl;
	
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++){
		if (fabs(yaw_rate) < EPS){
			particles[i].x += velocity*delta_t*(sin(particles[i].theta));
			particles[i].y += velocity*delta_t*(sin(particles[i].theta));
		}else{
			particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate*(-cos(particles[i].theta + yaw_rate*delta_t) + cos(particles[i].theta));
			particles[i].theta += yaw_rate/delta_t;
		}

		particles[i].x +=  dist_x(gen);
		particles[i].y +=  dist_y(gen);
		particles[i].theta +=  dist_theta(gen);
	}
	
	cout << "Done with predict" << endl;
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i = 0; i < observations.size(); i++){

		double min = numeric_limits<double>::max();


		for(int pred = 0; pred < predicted.size(); pred++){
			
			double dist = dist_2(predicted[pred].x, predicted[pred].y, observations[i].x, observations[i].y);
			if(dist < min){
				min = dist;
				observations[i].id = predicted[pred].id;
			}


		}
		 
	}

}


double get_gaus_weight(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y){
	// calculate normalization term
	double gauss_norm= (1.0/(2.0 * M_PI * sig_x * sig_y));

	// calculate exponent
	double exponent= ((x_obs - mu_x)*(x_obs - mu_x))/(2 * sig_x*sig_x) + ((y_obs - mu_y)*(y_obs - mu_y))/(2 * sig_y*sig_y);

	// calculate weight using normalization terms and exponent
	double weight= gauss_norm * exp(-exponent);
	return weight;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	
	//Loop over all particles 
	for(int i = 0; i < num_particles; i++){

		//Vector of landmarks within sensor range
		vector<LandmarkObs> predictions;
		for(int land = 0 ; land < map_landmarks.landmark_list.size(); land ++){
			float lx = map_landmarks.landmark_list[land].x_f;
      		float ly = map_landmarks.landmark_list[land].y_f;
			int id = map_landmarks.landmark_list[land].id_i;

			double range_x = lx - particles[i].x;
			double range_y = ly - particles[i].x;

			//Add to list of in-range landmarks
			if((range_x*range_x + range_y*range_y) <= sensor_range*sensor_range){
				LandmarkObs landmark;
				landmark.id = id;
				landmark.x = lx; 
				landmark.y = ly;
				predictions.push_back(landmark);
			}
		}


	
		//Transform to Map-space
		vector<LandmarkObs> observations_map;
		for(int j=0; j<observations.size(); j++){
			LandmarkObs landmark;
			landmark.x = (particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y));
			landmark.y = (particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y));
			observations_map.push_back(landmark);
		}
		
		//Associate observations to landmarks
		dataAssociation(predictions, observations_map);

		
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		
		// Calculate particle weight
		double weight = 1.0;
		for(int j=0; j<observations_map.size(); j++){
			// get the x,y coordinates of the landmark
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == observations_map[j].id) {
					weight = get_gaus_weight(std_landmark[0], std_landmark[1], observations_map[j].x, observations_map[j].y, predictions[k].x, predictions[k].y);
					break;
				}
			}
			
			// Handle if weight is 0
			if (weight < EPS) {
				particles[i].weight *= EPS;
				weights[i] = particles[i].weight;
			}else {
				particles[i].weight *= weight;
				weights[i] = particles[i].weight;
			}

			
			//Save result
			associations.push_back(observations_map[j].id);
			sense_x.push_back(observations_map[j].x);
			sense_y.push_back(observations_map[j].y);
			
		}

		//Sets the associations of that particle
		//particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

	}
	
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	
	default_random_engine gen;

	weights.clear();
	for(int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}
	discrete_distribution<int> particle_dist(weights.begin(),weights.end());

	// Resample
	vector<Particle> updated_particles;
	updated_particles.resize(num_particles);
	for(int i=0; i<num_particles; i++){
		auto index = particle_dist(gen);
		updated_particles[i] = move(particles[index]);
		
	}
	particles = move(updated_particles);
	
	
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

*/