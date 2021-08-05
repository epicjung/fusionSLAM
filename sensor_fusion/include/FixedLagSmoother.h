/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    FixedLagSmoother.h
 * @brief   Base class for a fixed-lag smoother. This mimics the basic interface to iSAM2.
 *
 * @author  Stephen Williams
 * @date    Feb 27, 2013
 */

// \callgraph
#pragma once

#include <gtsam_unstable/dllexport.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <map>
#include <vector>

namespace gtsam {

class FixedLagSmoother2 {

public:

  /// Typedef for a shared pointer to an Incremental Fixed-Lag Smoother
  typedef boost::shared_ptr<FixedLagSmoother2> shared_ptr;

  /// Typedef for a Key-Timestamp map/database
  typedef std::map<Key, double> KeyTimestampMap;
  typedef std::multimap<double, Key> TimestampKeyMap;

  /**
   * Meta information returned about the update
   */
  // TODO: Think of some more things to put here
  struct Result {
    size_t iterations; ///< The number of optimizer iterations performed
    size_t intermediateSteps; ///< The number of intermediate steps performed within the optimization. For L-M, this is the number of lambdas tried.
    size_t nonlinearVariables; ///< The number of variables that can be relinearized
    size_t linearVariables; ///< The number of variables that must keep a constant linearization point
    double error; ///< The final factor graph error
    std::map<Key, noiseModel::Gaussian::shared_ptr> marginalCovariances; 
    Result() : iterations(0), intermediateSteps(0), nonlinearVariables(0), linearVariables(0), error(0) {};

    /// Getter methods
    size_t getIterations() const { return iterations; }
    size_t getIntermediateSteps() const { return intermediateSteps; }
    size_t getNonlinearVariables() const { return nonlinearVariables; }
    size_t getLinearVariables() const { return linearVariables; }
    double getError() const { return error; }
    void print() const;
  };

  /** default constructor */
  FixedLagSmoother2(double smootherLag = 0.0) : smootherLag_(smootherLag) { }

  /** destructor */
  virtual ~FixedLagSmoother2(){}

  /** Print the factor for debugging and testing (implementing Testable) */
  virtual void print( const std::string& s = "FixedLagSmoother2:\n", const KeyFormatter& keyFormatter = DefaultKeyFormatter)
  {
    std::cout << s;
    std::cout << "  smoother lag: " << smootherLag_ << std::endl;
  };

//   /** Check if two IncrementalFixedLagSmoother2 Objects are equal */
  virtual bool equals(const FixedLagSmoother2& rhs, double tol = 1e-9) 
  {
    return std::abs(smootherLag_ - rhs.smootherLag_) < tol
      && std::equal(timestampKeyMap_.begin(), timestampKeyMap_.end(), rhs.timestampKeyMap_.begin());
  };

  /** read the current smoother lag */
  double smootherLag() const {
    return smootherLag_;
  }

  /** write to the current smoother lag */
  double& smootherLag() {
    return smootherLag_;
  }

  /** Access the current set of timestamps associated with each variable */
  const KeyTimestampMap& timestamps() const {
    return keyTimestampMap_;
  }

  /** Add new factors, updating the solution and relinearizing as needed. */
  virtual Result update(const NonlinearFactorGraph& newFactors = NonlinearFactorGraph(),
                        const Values& newTheta = Values(),
                        const KeyTimestampMap& timestamps = KeyTimestampMap(),
                        const FactorIndices& factorsToRemove = FactorIndices()) = 0;

  /** Compute an estimate from the incomplete linear delta computed during the last update.
   * This delta is incomplete because it was not updated below wildfire_threshold.  If only
   * a single variable is needed, it is faster to call calculateEstimate(const KEY&).
   */
  virtual Values calculateEstimate() const  = 0;


protected:

  /** The length of the smoother lag. Any variable older than this amount will be marginalized out. */
  double smootherLag_;

//   /** The current timestamp associated with each tracked key */
  TimestampKeyMap timestampKeyMap_;
  KeyTimestampMap keyTimestampMap_;

  /** Update the Timestamps associated with the keys */
  void updateKeyTimestampMap(const KeyTimestampMap& timestamps)
  {
    printf("updateKeyTimestampMap: Keys\n");
    // Loop through each key and add/update it in the map
    for(const auto& key_timestamp: timestamps) {
      std::cout << DefaultKeyFormatter(key_timestamp.first) << " ";
      // printf("Key: %d\n", key_timestamp.first);
      // Check to see if this key already exists in the database
      KeyTimestampMap::iterator keyIter = keyTimestampMap_.find(key_timestamp.first);

      // If the key already exists
      if(keyIter != keyTimestampMap_.end()) {
        printf("Already Exists\n");
        // Find the entry in the Timestamp-Key database
        std::pair<TimestampKeyMap::iterator,TimestampKeyMap::iterator> range = timestampKeyMap_.equal_range(keyIter->second);
        TimestampKeyMap::iterator timeIter = range.first;
        while(timeIter->second != key_timestamp.first) {
          ++timeIter;
        }
        // remove the entry in the Timestamp-Key database
        timestampKeyMap_.erase(timeIter);
        // insert an entry at the new time
        timestampKeyMap_.insert(TimestampKeyMap::value_type(key_timestamp.second, key_timestamp.first));
        // update the Key-Timestamp database
        keyIter->second = key_timestamp.second;
      } else {
        printf("New\n");
        // Add the Key-Timestamp database
        keyTimestampMap_.insert(key_timestamp);
        // Add the key to the Timestamp-Key database
        timestampKeyMap_.insert(TimestampKeyMap::value_type(key_timestamp.second, key_timestamp.first));
      }
    }
  }

  /** Erase keys from the Key-Timestamps database */
  void eraseKeyTimestampMap(const KeyVector& keys)
  {
    for(Key key: keys) {
      // Erase the key from the Timestamp->Key map
      double timestamp = keyTimestampMap_.at(key);

      TimestampKeyMap::iterator iter = timestampKeyMap_.lower_bound(timestamp);
      while(iter != timestampKeyMap_.end() && iter->first == timestamp) {
        if(iter->second == key) {
          timestampKeyMap_.erase(iter++);
        } else {
          ++iter;
        }
      }
        // Erase the key from the Key->Timestamp map
      keyTimestampMap_.erase(key);
    }
  }

  /** Find the most recent timestamp of the system */
  double getCurrentTimestamp() const
  {
    if(timestampKeyMap_.size() > 0) {
      return timestampKeyMap_.rbegin()->first;
    } else {
      return -std::numeric_limits<double>::max();
    }
  }

  /** Find all of the keys associated with timestamps before the provided time */
  KeyVector findKeysBefore(double timestamp) const
  {
    KeyVector keys;
    TimestampKeyMap::const_iterator end = timestampKeyMap_.lower_bound(timestamp);
    for(TimestampKeyMap::const_iterator iter = timestampKeyMap_.begin(); iter != end; ++iter) {
      keys.push_back(iter->second);
    }
    return keys;
  }

  /** Find all of the keys associated with timestamps before the provided time */
  KeyVector findKeysAfter(double timestamp) const
  {
    KeyVector keys;
    TimestampKeyMap::const_iterator begin = timestampKeyMap_.upper_bound(timestamp);
    for(TimestampKeyMap::const_iterator iter = begin; iter != timestampKeyMap_.end(); ++iter) {
      keys.push_back(iter->second);
    }
    return keys;
  }

}; // FixedLagSmoother2

// /// Typedef for matlab wrapping
typedef FixedLagSmoother2::KeyTimestampMap FixedLagSmootherKeyTimestampMap;
// typedef FixedLagSmootherKeyTimestampMap::value_type FixedLagSmootherKeyTimestampMapValue;
// typedef FixedLagSmoother::Result FixedLagSmootherResult;

} /// namespace gtsam