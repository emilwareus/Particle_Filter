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

		particles.push_back(part);				
		 
	}

}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 50;	

	gaussian_init(x, y, theta, std[0], std[1], std[2], num_particles);
	
	bool is_initialized = true;
	
	cout << "Initialized!"<< endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	cout << "Let's Predict!"<< endl;
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

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

	cout << "Prediction done!"<< endl;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i = 0; i < observations.size(); i++){

		double min = numeric_limits<double>::max();
		int id = 0;

		for(int pred = 0; pred < predicted.size(); pred++){
			
			double dist = sqrt((observations[i].x - predicted[pred].x) * (observations[i].x - predicted[pred].x) + (observations[i].y- predicted[pred].y)*(observations[i].y- predicted[pred].y));

			if(dist < min){
				min = dist;
				id = predicted[pred].id;
			}


		}
		observations[i].id = id; 
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


	cout << "Let's update the weights" << endl;
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
			if(range_x*range_x + range_y*range_y <= sensor_range*sensor_range){
				predictions.push_back(LandmarkObs{ id, lx, lx });
			}
		}


		//Transform to Map-space
		vector<LandmarkObs> observations_map;
		for(int j=0; j<observations.size(); j++){
			observations_map.push_back(LandmarkObs{ observations[j].id, 
						(particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y)),
						(particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y))});
		}
		
		//Associate observations to landmarks
		dataAssociation(predictions, observations_map);

		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		// Calculate particle weight
		for(int j=0; j<observations_map.size(); j++){
			// get the x,y coordinates of the landmark
			double weight = 0.0;
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == observations_map[j].id) {
					weight = get_gaus_weight(std_landmark[0], std_landmark[0], observations_map[j].x, observations_map[j].y, predictions[k].x, predictions[k].y);
					break;
				}
			}
			
			// Handle if weight is 0
			if (weight < EPS) {
				particles[i].weight *= EPS;
			}else {
				particles[i].weight *= weight;
			}

			//Save result
			associations.push_back(observations_map[j].id);
			sense_x.push_back(observations_map[j].x);
			sense_y.push_back(observations_map[j].y);

		}

		//Sets the associations of that particle
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

	}
	cout << "Weights done!"<< endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	cout << "What if we resample?" << endl;
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
	particles = updated_particles;
	cout << "Resampeling done!" << endl;
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
