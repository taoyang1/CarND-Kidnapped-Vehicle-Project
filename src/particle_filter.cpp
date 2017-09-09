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

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	num_particles = 1000;
	weights.resize(num_particles);
	for (int i = 0; i < num_particles; ++i) {

		Particle particle;

		double sample_x, sample_y, sample_theta;
		
		// TODO: Sample  and from these normal distrubtions like this: 
		//	 sample_x = dist_x(gen);
		//	 where "gen" is the random engine initialized earlier.
		
		 sample_x = dist_x(gen);
		 sample_y = dist_y(gen);
		 sample_theta = dist_theta(gen);	 
		 
		 particle.id = i;
		 particle.x = sample_x;
		 particle.y = sample_y;
		 particle.theta = sample_theta;
		 particle.weight = 1.0;
		 weights[i] = particle.weight;

		 // push the sampled particle to the vector
		 particles.push_back(particle);
	}
	
	is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  	// define normal distributions for sensor noise
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);


	// first evolve the particle according to the bicycle model
	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.00001) {
			// bicycle model equation for yaw_rate = 0
			particles[i].x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			particles[i].y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;

		} else {
			particles[i].x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t)- sin(particles[i].theta));
			particles[i].y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta)- cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;	
		}		

	    // add noise
	    particles[i].x += N_x(gen);
	    particles[i].y += N_y(gen);
	    particles[i].theta += N_theta(gen);
	}
}

// void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
// 	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
// 	//   observed measurement to this particular landmark.
// 	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
// 	//   implement this method and use it as a helper during the updateWeights phase.

// }

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& nn_landmarks) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	int num_observations = predicted.size();
	int num_landmarks = nn_landmarks.size();

	vector<LandmarkObs> tmp_landmark = nn_landmarks;

	for (int i=0;i<num_observations;i++) {

		double min_distance = numeric_limits<double>::max();
		int nn_index = -1;

		for (int j=0;j<num_landmarks;j++) {
			double cur_distance = dist(predicted[i].x,predicted[i].y,tmp_landmark[j].x,tmp_landmark[j].y);
			if (cur_distance < min_distance) {
				min_distance = cur_distance;
				nn_index = j;
			}
		}

		nn_landmarks[i].id = tmp_landmark[nn_index].id;
		nn_landmarks[i].x = tmp_landmark[nn_index].x;
		nn_landmarks[i].y = tmp_landmark[nn_index].y;

	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	int num_observations = observations.size();
	
	double gauss_norm, exponent;
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	for (int i = 0; i<num_particles; i++) {

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// landmarks are within the sensor range
		vector<LandmarkObs> nearest_landmarks;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      		// get id and x,y coordinates
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

      		// use rectangular region for computational efficiency
			if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range) {
				nearest_landmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });	
			}
		}

		vector<LandmarkObs> predicted_observations(num_observations);

		// transofrm the local observation of landmarks to map coordinates
		for (int j = 0; j<num_observations;j++) {
			predicted_observations[j].id = observations[j].id;
			predicted_observations[j].x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
			predicted_observations[j].y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
			// cout << predicted_observations[j].x << predicted_observations[j].y << endl;
		}
		

		// find the nearest landmark wrt the particle observation in map coordinates
		// this will return the nn_landmark list which corresponds to the predicted observations
		// dataAssociation(predicted_observations,nearest_landmarks, map_landmarks.landmark_list);
		dataAssociation(predicted_observations, nearest_landmarks);

		// finally, calculate the weights	
		weights[i] = 1.0;
		for (int j = 0; j<num_observations; j++) {
			gauss_norm = 1/(2.0 * M_PI * sig_x * sig_y);
			exponent = pow((predicted_observations[j].x - nearest_landmarks[j].x),2.0)/(2 * sig_x * sig_x) + pow((predicted_observations[j].y - nearest_landmarks[j].y),2.0)/(2 * sig_y * sig_y);
			weights[i] *= gauss_norm * exp(-exponent);
		}
		// cout << weights[i] <<endl;

	}

	// normalize weights
	// std::vector<double> tempWeights;
	weights = normalize_vector(weights);

	for (int i = 0; i<num_particles; i++) {
		// cout << tempWeights[i] << endl;
		particles[i].weight = weights[i];
		// cout << weights[i] << endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// resample the particles according to the weights
	// default_random_engine gen;
	std::discrete_distribution<> d(weights.begin(),weights.end());
	// std::discrete_distribution<> d(weights);


	vector<Particle> temp_particles = particles;
	for (int i = 0; i<num_particles; i++) {
		particles[i] = temp_particles[d(gen)];
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
