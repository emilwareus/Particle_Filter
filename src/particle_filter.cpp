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






void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 50;	

	
	std::default_random_engine gen;
	// Standard deviations for x, y, and theta

	// This line creates a normal (Gaussian) distribution for x, y and theta
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {

		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);			
			
	}

	
	is_initialized = true;
	
	cout << "Initialized!"<< endl;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	cout << "Prediction" << endl;
	std::default_random_engine gen;
   


	for(int i = 0; i < num_particles; i++){

		double x;
		double y;
		double theta;
		if (yaw_rate){
			x = particles[i].x + velocity / yaw_rate *(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			y = particles[i].y + velocity / yaw_rate *(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			theta = particles[i].theta + yaw_rate * delta_t;
			
		}else{
			x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			theta = particles[i].theta;
		}

		
		std::normal_distribution<double> dist_x(x, std_pos[0]);
		std::normal_distribution<double> dist_y(y, std_pos[1]);
		std::normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x =  dist_x(gen);
		particles[i].y =  dist_y(gen);
		particles[i].theta =  dist_theta(gen);
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

int index(std::vector<LandmarkObs> predicted, int id) {
  for (int i = 0; i < predicted.size(); ++i) {
    if (predicted[i].id == id) {
      return i;
    }
  }
  return -1;
}

double gauss(double x, double y, double lm_x, double lm_y, double std_x,
                   double std_y) {
  double a = pow(x - lm_x, 2.0) / (2.0 * pow(std_x, 2.0));
  double b = pow(y - lm_y, 2.0) / (2.0 * pow(std_y, 2.0));
  double p = exp(-(a + b)) / (2.0 * M_PI * std_x * std_y);
  
  return p;
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
		for(int j = 0 ; j < map_landmarks.landmark_list.size(); j ++){
			double range = dist(particles[i].x, particles[i].y,
               map_landmarks.landmark_list[j].x_f,
               map_landmarks.landmark_list[j].y_f);
			//Add to list of in-range landmarks
			if(range <= sensor_range){
				LandmarkObs landmark;
				landmark.id = map_landmarks.landmark_list[j].id_i;
				landmark.x = map_landmarks.landmark_list[j].x_f;
				landmark.y = map_landmarks.landmark_list[j].y_f;
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

		
		
		// Calculate particle weight
		double weight = 1.0;
		for(int j=0; j<observations_map.size(); j++){
			
			int lm_index = index(predictions, observations_map[j].id);
			double l_x = predictions[lm_index].x;
			double l_y = predictions[lm_index].y;
			weight *= gauss(observations_map[j].x, observations_map[j].y,
									l_x, l_y, std_landmark[0], std_landmark[1]);
		}
			

		particles[i].weight = weight;
		weights[i] = weight;
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

