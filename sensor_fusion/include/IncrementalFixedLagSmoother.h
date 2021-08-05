/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    IncrementalFixedLagSmoother.h
 * @brief   An iSAM2-based fixed-lag smoother.
 *
 * @author  Michael Kaess, Stephen Williams
 * @date    Oct 14, 2012
 */

// \callgraph
#pragma once

#include "FixedLagSmoother.h"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/base/debug.h>

namespace gtsam {

/**
 * This is a base class for the various HMF2 implementations. The HMF2 eliminates the factor graph
 * such that the active states are placed in/near the root. This base class implements a function
 * to calculate the ordering, and an update function to incorporate new factors into the HMF.
 */
class IncrementalFixedLagSmoother2: public FixedLagSmoother2 {

public:

  /// Typedef for a shared pointer to an Incremental Fixed-Lag Smoother
  typedef boost::shared_ptr<IncrementalFixedLagSmoother2> shared_ptr;

  /** default constructor */
  IncrementalFixedLagSmoother2(double smootherLag = 0.0,
      const ISAM2Params& parameters = DefaultISAM2Params()) :
      FixedLagSmoother2(smootherLag), isam_(parameters) {
  }

  /** destructor */
  ~IncrementalFixedLagSmoother2() {
      std::cout << "IncermentalFixedLag destructor" << std::endl;
  }

//   /** Print the factor for debugging and testing (implementing Testable) */
  void print(const std::string& s = "IncrementalFixedLagSmoother2:\n",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter){
            FixedLagSmoother2::print(s, keyFormatter);
      }

//   /** Check if two IncrementalFixedLagSmoother Objects are equal */
  bool equals(const FixedLagSmoother2& rhs, double tol = 1e-9)
  {
        const IncrementalFixedLagSmoother2* e = dynamic_cast<const IncrementalFixedLagSmoother2*>(&rhs);
        return e != nullptr && FixedLagSmoother2::equals(*e, tol)
      && isam_.equals(e->isam_, tol);
  }

  void recursiveMarkAffectedKeys(const Key& key,
    const ISAM2Clique::shared_ptr& clique, std::set<Key>& additionalKeys) {

    // Check if the separator keys of the current clique contain the specified key
    if (std::find(clique->conditional()->beginParents(),
        clique->conditional()->endParents(), key)
        != clique->conditional()->endParents()) {

      // Mark the frontal keys of the current clique
      for(Key i: clique->conditional()->frontals()) {
        additionalKeys.insert(i);
      }

      // Recursively mark all of the children
      for(const ISAM2Clique::shared_ptr& child: clique->children) {
        recursiveMarkAffectedKeys(key, child, additionalKeys);
      }
    }
    // If the key was not found in the separator/parents, then none of its children can have it either
  }


/* ************************************************************************* */
/** Erase any keys associated with timestamps before the provided time */

void eraseKeysBefore(double timestamp) {
  TimestampKeyMap::iterator end = timestampKeyMap_.lower_bound(timestamp);
  TimestampKeyMap::iterator iter = timestampKeyMap_.begin();
  while (iter != end) {
    keyTimestampMap_.erase(iter->second);
    timestampKeyMap_.erase(iter++);
  }
}

/* ************************************************************************* */
void createOrderingConstraints(
    const KeyVector& marginalizableKeys,
    boost::optional<FastMap<Key, int> >& constrainedKeys) const {
  if (marginalizableKeys.size() > 0) {
    constrainedKeys = FastMap<Key, int>();
    // Generate ordering constraints so that the marginalizable variables will be eliminated first
    // Set all variables to Group1
    for(const TimestampKeyMap::value_type& timestamp_key: timestampKeyMap_) {
      constrainedKeys->operator[](timestamp_key.second) = 1;
    }
    // Set marginalizable variables to Group0
    for(Key key: marginalizableKeys) {
      constrainedKeys->operator[](key) = 0;
    }
  }
}

/* ************************************************************************* */
void getMargianlCovariances(const KeyVector& marginalizableKeys, 
                            std::map<Key, gtsam::noiseModel::Gaussian::shared_ptr> &marginalCovariances) const
{
  if (marginalizableKeys.size() > 0)
  {
    for(Key key : marginalizableKeys)
    {
      gtsam::noiseModel::Gaussian::shared_ptr marginalCov = gtsam::noiseModel::Gaussian::Covariance(isam_.marginalCovariance(key));
      marginalCovariances[key] = marginalCov;
    }
  }
}

/* ************************************************************************* */
void PrintKeySet(const std::set<Key>& keys,
    const std::string& label) {
  std::cout << label;
  for(Key key: keys) {
    std::cout << " " << DefaultKeyFormatter(key);
  }
  std::cout << std::endl;
}

/* ************************************************************************* */
void PrintSymbolicFactor(
    const GaussianFactor::shared_ptr& factor) {
  std::cout << "f(";
  for(Key key: factor->keys()) {
    std::cout << " " << DefaultKeyFormatter(key);
  }
  std::cout << " )" << std::endl;
}

/* ************************************************************************* */
void PrintSymbolicGraph(
    const GaussianFactorGraph& graph, const std::string& label) {
  std::cout << label << std::endl;
  for(const GaussianFactor::shared_ptr& factor: graph) {
    PrintSymbolicFactor(factor);
  }
}

/* ************************************************************************* */
void PrintSymbolicTree(const ISAM2& isam,
    const std::string& label) {
  std::cout << label << std::endl;
  if (!isam.roots().empty()) {
    for(const ISAM2::sharedClique& root: isam.roots()) {
      PrintSymbolicTreeHelper(root);
    }
  } else
    std::cout << "{Empty Tree}" << std::endl;
}

/* ************************************************************************* */
void PrintSymbolicTreeHelper(
    const ISAM2Clique::shared_ptr& clique, const std::string indent = "") {

  // Print the current clique
  std::cout << indent << "P( ";
  for(Key key: clique->conditional()->frontals()) {
    std::cout << DefaultKeyFormatter(key) << " ";
  }
  if (clique->conditional()->nrParents() > 0)
    std::cout << "| ";
  for(Key key: clique->conditional()->parents()) {
    std::cout << DefaultKeyFormatter(key) << " ";
  }
  std::cout << ")" << std::endl;

  // Recursively print all of the children
  for(const ISAM2Clique::shared_ptr& child: clique->children) {
    PrintSymbolicTreeHelper(child, indent + " ");
  }
}

//   /**
//    * Add new factors, updating the solution and re-linearizing as needed.
//    * @param newFactors new factors on old and/or new variables
//    * @param newTheta new values for new variables only
//    * @param timestamps an (optional) map from keys to real time stamps
//    * @param factorsToRemove an (optional) list of factors to remove.
//    */
  Result update(const NonlinearFactorGraph& newFactors = NonlinearFactorGraph(),
                const Values& newTheta = Values(), //
                const KeyTimestampMap& timestamps = KeyTimestampMap(),
                const FactorIndices& factorsToRemove = FactorIndices())
  {

    // const bool debug = ISDEBUG("IncrementalFixedLagSmoother update");
    const bool debug = true;
    if (debug) {
        std::cout << "IncrementalFixedLagSmoother::update() Start" << std::endl;
        PrintSymbolicTree(isam_, "Bayes Tree Before Update:");
        std::cout << "END" << std::endl;
    }

    FastVector<size_t> removedFactors;
    boost::optional<FastMap<Key, int> > constrainedKeys = boost::none;
    std::map<Key, gtsam::noiseModel::Gaussian::shared_ptr> marginalCovariances;

    // Update the Timestamps associated with the factor keys
    updateKeyTimestampMap(timestamps);

    // Get current timestamp
    double current_timestamp = getCurrentTimestamp();

    if (debug)
        std::cout << "Current Timestamp: " << current_timestamp << std::endl;

    // Find the set of variables to be marginalized out
    KeyVector marginalizableKeys = findKeysBefore(
        current_timestamp - smootherLag_);

    if (debug) {
        std::cout << "Marginalizable Keys: ";
        for(Key key: marginalizableKeys) {
        std::cout << DefaultKeyFormatter(key) << " ";
        }
        std::cout << std::endl;
    }

    // Force iSAM2 to put the marginalizable variables at the beginning
    createOrderingConstraints(marginalizableKeys, constrainedKeys);

    if (debug) {
        std::cout << "Constrained Keys: ";
        if (constrainedKeys) {
        for (FastMap<Key, int>::const_iterator iter = constrainedKeys->begin();
            iter != constrainedKeys->end(); ++iter) {
            std::cout << DefaultKeyFormatter(iter->first) << "(" << iter->second
                << ")  ";
        }
        }
        std::cout << std::endl;
    }

    // Get margianl covariance
    getMargianlCovariances(marginalizableKeys, marginalCovariances);

    // Mark additional keys between the marginalized keys and the leaves
    std::set<Key> additionalKeys;
    // auto &factorGraph = isam_.getFactorsUnsafe();
    // factorGraph.print("************Current factor graphs************\n");
    for(Key key: marginalizableKeys) {
      ISAM2Clique::shared_ptr clique = isam_[key];
      for(const ISAM2Clique::shared_ptr& child: clique->children) {
        recursiveMarkAffectedKeys(key, child, additionalKeys);
      }
    }
    KeyList additionalMarkedKeys(additionalKeys.begin(), additionalKeys.end());

    // Update iSAM2
    printf("additionKeys size: %d\n", additionalMarkedKeys.size());
    for (Key key : additionalMarkedKeys)
      std::cout << DefaultKeyFormatter(key) << " ";
    std::cout << std::endl;
    printf("factorsToRemove size: %d\n", factorsToRemove.size());
    newFactors.print("*******New Factors**********: \n");
    SETDEBUG("ISAM2 update", true);
    
    isamResult_ = isam_.update(newFactors, newTheta,
        factorsToRemove, constrainedKeys, boost::none, additionalMarkedKeys);
    if (debug) {
        PrintSymbolicTree(isam_,
            "Bayes Tree After Update, Before Marginalization:");
        std::cout << "END" << std::endl;
    }

    // Marginalize out any needed variables
    if (marginalizableKeys.size() > 0) {
        FastList<Key> leafKeys(marginalizableKeys.begin(),
            marginalizableKeys.end());
        isam_.marginalizeLeaves(leafKeys);
    }

    // Remove marginalized keys from the KeyTimestampMap
    eraseKeyTimestampMap(marginalizableKeys);

    if (debug) {
        PrintSymbolicTree(isam_, "Final Bayes Tree:");
        std::cout << "END" << std::endl;
    }

    // TODO: Fill in result structure
    Result result;
    result.iterations = 1;
    result.linearVariables = 0;
    result.nonlinearVariables = 0;
    result.error = 0;
    result.marginalCovariances = marginalCovariances;
    if (debug)
        std::cout << "IncrementalFixedLagSmoother::update() Finish" << std::endl;

    return result;
  }

  /** Compute an estimate from the incomplete linear delta computed during the last update.
   * This delta is incomplete because it was not updated below wildfire_threshold.  If only
   * a single variable is needed, it is faster to call calculateEstimate(const KEY&).
   */
  Values calculateEstimate() const override {
    return isam_.calculateEstimate();
  }

  /** Compute an estimate for a single variable using its incomplete linear delta computed
   * during the last update.  This is faster than calling the no-argument version of
   * calculateEstimate, which operates on all variables.
   * @param key
   * @return
   */
  template<class VALUE>
  VALUE calculateEstimate(Key key) const {
    return isam_.calculateEstimate<VALUE>(key);
  }

  /** return the current set of iSAM2 parameters */
  const ISAM2Params& params() const {
    return isam_.params();
  }

  /** Access the current set of factors */
  const NonlinearFactorGraph& getFactors() const {
    return isam_.getFactorsUnsafe();
  }

  /** Access the current linearization point */
  const Values& getLinearizationPoint() const {
    return isam_.getLinearizationPoint();
  }

  /** Access the current set of deltas to the linearization point */
  const VectorValues& getDelta() const {
    return isam_.getDelta();
  }

  /// Calculate marginal covariance on given variable
  Matrix marginalCovariance(Key key) const {
    return isam_.marginalCovariance(key);
  }

  /// Get results of latest isam2 update
  const ISAM2Result& getISAM2Result() const{ return isamResult_; }

protected:

  /** Create default parameters */
  static ISAM2Params DefaultISAM2Params() {
    ISAM2Params params;
    params.findUnusedFactorSlots = true;
    return params;
  }

  /** An iSAM2 object used to perform inference. The smoother lag is controlled
   * by what factors are removed each iteration */
  ISAM2 isam_;

  /** Store results of latest isam2 update */
  ISAM2Result isamResult_;

private:
  // /** Private methods for printing debug information */
  // static void PrintKeySet(const std::set<Key>& keys, const std::string& label =
  //     "Keys:");
  // static void PrintSymbolicFactor(const GaussianFactor::shared_ptr& factor);
  // static void PrintSymbolicGraph(const GaussianFactorGraph& graph,
  //     const std::string& label = "Factor Graph:");
  // static void PrintSymbolicTree(const gtsam::ISAM2& isam,
  //     const std::string& label = "Bayes Tree:");
  // static void PrintSymbolicTreeHelper(
  //     const gtsam::ISAM2Clique::shared_ptr& clique, const std::string indent =
  //         "");

};
// IncrementalFixedLagSmoother

}/// namespace gtsam