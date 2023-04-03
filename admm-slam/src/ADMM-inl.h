#include <ADMM.h>

#define USE_L2_NORM 1

using namespace gtsam;

/********************************************************************************************/
template <class POSE_TYPE>
void ADMM<POSE_TYPE>::logResult(int iter, double p_res, double d_res, double time_taken) {
  // Create directories if they do not exist to save intermediate graphs
  logDir_ = outputDir_ + ("/intermediateGraphs/");
  if (!(boost::filesystem::exists(logDir_))) {
    boost::filesystem::create_directory(logDir_);
  }

  subgraphLogDir_ = outputDir_ + ("/intermediateSubGraphs/");
  if (!(boost::filesystem::exists(subgraphLogDir_))) {
    boost::filesystem::create_directory(subgraphLogDir_);
  }

  // Pack results into a single Value structure
  Values intermediateResult;
  for (size_t i = 0; i < subinitials_.size(); i++) {
    for (const Values::ConstKeyValuePair &keyValue : subinitials_[i]) {
      if (!intermediateResult.exists(keyValue.key)) intermediateResult.insert(keyValue.key, keyValue.value);
    }

    if (logCount_ > logCountThresh_) {
      std::string subgraph_filepath =
          subgraphLogDir_ + boost::lexical_cast<std::string>(iter) + "_" + boost::lexical_cast<std::string>(i) + ".g2o";
      // writeG2o(subgraphs_[i], subinitials_[i], subgraph_filepath);
    }
  }

  if (logCount_ > logCountThresh_) {
    std::string filepath = logDir_ + boost::lexical_cast<std::string>(iter) + ".g2o";
    // writeG2o(fullGraph_, intermediateResult, filepath);
    logCount_ = 0;
  }

  double admmError = 2 * fullGraph_.error(intermediateResult);
  *residualLog_ << iter << ", " << p_res << ", " << d_res << ", " << rho_ << ", " << admmError << "," << time_taken
                << std::endl;

  if (verbosity_ >= DEBUG) std::cout << "ADMM Error: " << admmError << std::endl;
}

/********************************************************************************************/
// Flagged Initialization
template <class POSE_TYPE>
void ADMM<POSE_TYPE>::FlaggedInitialization() {
  GaussNewtonParams params;
  params.setVerbosity("ERROR");
  params.setRelativeErrorTol(-1e+10);
  params.setAbsoluteErrorTol(-1e+10);
  params.setMaxIterations(3);
  int num_subgraphs = subgraphs_.size();
  int nrSeparators = separators_.size() / 3;

  SharedNoiseModel rhoNoiseModel = noiseModel::Isotropic::Variance(dualDimension_, 2 / rho_);

  std::vector<Values> initial_old(subinitials_.size(), Values());
  for (int sub_id = 0; sub_id < num_subgraphs; sub_id++) initial_old[sub_id] = Values(subinitials_[sub_id]);

  std::vector<int> initFlag(
      subinitials_.size(),
      0);  // 0: subgraph is not optimized yet, 1: subgraph to be optimized in next iteration, 2: subgraph optimized
  initFlag[0] = 1;

  /// Keep optimizing un-initialized subgraphs
  while (1) {
    bool optimizedSomething = false;  // check if it has optimized atelast one sub-graph
    for (int sub_id = 0; sub_id < num_subgraphs; sub_id++) {
      if (initFlag[sub_id] == 1) {
        if (verbosity_ >= DEBUG) printf("Flagged Initialization of %d\n", sub_id);
        optimizedSomething = true;

        // Create the local problem
        NonlinearFactorGraph sub_graph = subgraphs_[sub_id];
        Values sub_initial = initial_old[sub_id];

        // add penalty factors for each constraint (i.e., for each separator)
        for (int i = 0; i < nrSeparators; i++) {
          if (separators_[3 * i + 1] == sub_id) {  // behaves like f_x
            Key key = separators_[3 * i];          // separator node symbol in the current subgraph

            int graph_index = separators_[3 * i + 2];  // index of the neighboring subgraph

            if (initFlag[graph_index] ==
                2) {  // Add the BiasedPriorFactor if the neighboring graph is already initialized
              Vector u_k(dualDimension_);
              for (size_t k = 0; k < dualDimension_; k++) {
                u_k(k) = uall_k[dualDimension_ * i + k];
              }
              POSE_TYPE pose_z = initial_old[graph_index].at<POSE_TYPE>(key);
              sub_graph.add(BiasedPriorFactor<POSE_TYPE, Vector>(key, pose_z, -u_k, rhoNoiseModel));
            } else {
              initFlag[graph_index] = 1;  // Optimize it in the next iteration
            }
          } else if (separators_[3 * i + 2] == sub_id)  // behaves like g_z
          {
            Key key = separators_[3 * i];              // separator node symbol in the current subgraph
            int graph_index = separators_[3 * i + 1];  // index of the neighboring subgraph

            if (initFlag[graph_index] ==
                2) {  // Add the BiasedPriorFactor if the neighboring graph is already initialized

              Vector u_k(dualDimension_);
              for (size_t k = 0; k < dualDimension_; k++) {
                u_k(k) = uall_k[dualDimension_ * i + k];
              }
              POSE_TYPE pose_x = initial_old[graph_index].at<POSE_TYPE>(key);
              sub_graph.add(BiasedPriorFactor<POSE_TYPE, Vector>(key, pose_x, u_k, rhoNoiseModel));
            } else {
              initFlag[graph_index] = 1;  // Optimize it in the next iteration
            }
          }
        }

        if (solver_ == LAGO) {
          subinitials_[sub_id] = lago::initialize(sub_graph, false);
        } else if (solver_ == GN) {
          GaussNewtonOptimizer optimizer_subgraph(sub_graph, sub_initial, params);
          subinitials_[sub_id] = optimizer_subgraph.optimize();
        } else if (solver_ == LM) {
          std::cout << std::endl << "----------------" << std::endl;
          LevenbergMarquardtParams params;
          params.setlambdaInitial(1e-9);
          params.setVerbosityLM("TRYLAMBDA");

          LevenbergMarquardtOptimizer optimizer_subgraph(sub_graph, sub_initial, params);
          subinitials_[sub_id] = optimizer_subgraph.optimize();
          std::cout << std::endl << "----------------" << std::endl;
        } else {
          printf("ERROR: Specify solver type\n");
        }

        initFlag[sub_id] = 2;  // subGraph initialized
      }
    }

    if (!optimizedSomething) {  // Break if we haven't optimized anything
      if (verbosity_ >= DEBUG) printf("Done initializing\n");
      break;
    }
  }  // end of loop over subgraphs
}

/********************************************************************************************/
// ADMM Scaled Iteration
template <class POSE_TYPE>
std::pair<double, double> ADMM<POSE_TYPE>::iterate(int iter) {
  std::vector<VectorValues> subgraph_gradient(subgraphs_.size(), VectorValues());

  GaussNewtonParams params;
  if (iter == 0) {
    params.setRelativeErrorTol(-1e+10);
    params.setAbsoluteErrorTol(-1e+10);
    // params.setVerbosity("ERROR");
    params.setMaxIterations(3);
  }

  int num_subgraphs = subgraphs_.size();
  int nrSeparators = separators_.size() / 3;

  // The penalty parameter plays the role of an information content
  SharedNoiseModel rhoNoiseModel = noiseModel::Isotropic::Variance(dualDimension_, 2 / rho_);

  std::vector<Values> initial_old(subinitials_.size(), Values());
  for (int sub_id = 0; sub_id < num_subgraphs; sub_id++) initial_old[sub_id] = Values(subinitials_[sub_id]);

  /// Optimize each subgraph independently, but possibly sequentially (if parallel = false)
  for (int sub_id = 0; sub_id < num_subgraphs; sub_id++) {
    //    std::cout << "SUB ID " << sub_id << std::endl;
    // update sequentially
    if (!isParallel_) initial_old = subinitials_;  // we use the latest estimate for all subgraphs

    // Create the local problem
    NonlinearFactorGraph sub_graph = subgraphs_[sub_id];
    Values sub_initial = initial_old[sub_id];

    // if(sub_id != 0){
    // add penalty factors for each constraint (i.e., for each separator)
    for (int i = 0; i < nrSeparators; i++) {
      if (separators_[3 * i + 1] == sub_id) {  // behaves like f_x
        Key key = separators_[3 * i];          // separator node symbol in the current subgraph

        int graph_index = separators_[3 * i + 2];  // index of the neighboring subgraph
        Vector u_k(dualDimension_);
        std::cout << "U: ";
        for (size_t k = 0; k < dualDimension_; k++) {
          u_k(k) = uall_k[dualDimension_ * i + k];
          std::cout << u_k(k) <<", ";
        }
        std::cout << std::endl;
        POSE_TYPE pose_z = initial_old[graph_index].at<POSE_TYPE>(key);
        sub_graph.add(BiasedPriorFactor<POSE_TYPE, Vector>(key, pose_z, -u_k, rhoNoiseModel));
      } else if (separators_[3 * i + 2] == sub_id)  // behaves like g_z
      {
        Key key = separators_[3 * i];  // separator node symbol in the current subgraph

        int graph_index = separators_[3 * i + 1];  // index of the neighboring subgraph
        Vector u_k(dualDimension_);
        for (size_t k = 0; k < dualDimension_; k++) {
          u_k(k) = uall_k[dualDimension_ * i + k];
        }
        POSE_TYPE pose_x = initial_old[graph_index].at<POSE_TYPE>(key);
        sub_graph.add(BiasedPriorFactor<POSE_TYPE, Vector>(key, pose_x, u_k, rhoNoiseModel));
      }
    }

    if (solver_ == LAGO) {
      subinitials_[sub_id] = lago::initialize(sub_graph, false);
    } else if (solver_ == GN) {
      GaussNewtonOptimizer optimizer_subgraph(sub_graph, sub_initial, params);
      subinitials_[sub_id] = optimizer_subgraph.optimize();
    } else if (solver_ == LM) {
      LevenbergMarquardtParams params;
      params.setlambdaInitial(1e-7);
      LevenbergMarquardtOptimizer optimizer_subgraph(sub_graph, sub_initial, params);
      subinitials_[sub_id] = optimizer_subgraph.optimize();
    } else {
      printf("ERROR: Specify solver type\n");
    }

    // Subgraph gradient to be used later to compute dual residual
    subgraph_gradient[sub_id] = subgraphs_[sub_id].linearize(subinitials_[sub_id])->gradientAtZero();
  }  // end of loop over subgraphs

  // Dual update & computation of primal and dual residual for stopping conditions
  double p_res = 0, d_res = 0;

  NonlinearFactorGraph constraintsGraph;    // to compure d_res
  for (int i = 0; i < nrSeparators; i++) {  // for each separator
    Key key = separators_[3 * i];
    int sub_map1 = separators_[3 * i + 1];
    int sub_map2 = separators_[3 * i + 2];
    POSE_TYPE pose_x = subinitials_[sub_map1].at<POSE_TYPE>(key);
    POSE_TYPE pose_z = subinitials_[sub_map2].at<POSE_TYPE>(key);

    Matrix J_x, J_z;  // Jacobian matrix w.r.t x and z

    // (Dan) Below is origional, that computes the jacobians wrong
    // Pose2 between_x_z = pose_x.between(pose_z, J_x, J_z);
    // Vector b = Pose2::Logmap(between_x_z);
    Vector b = traits<POSE_TYPE>::Local(pose_x, pose_z, J_x, J_z);  // (Dan) Correct jacobians

    if (computeSubgraphGradient_) {
      // add the corresponding jacobians to the subgraph gradient of the corresponding subgraphs
      // 1/2 is used to compensate the definition of the cost function in GTSAM being multiplied by 1/2
      Vector y_k(dualDimension_);
      for (size_t k = 0; k < dualDimension_; k++) {
        y_k(k) = uall_k[dualDimension_ * i + k];
      }
      Vector scaled_Jx = y_k.transpose() * J_x;
      Vector scaled_Jz = y_k.transpose() * J_z;

      if (subgraph_gradient[sub_map1].exists(key)) {
        subgraph_gradient[sub_map1].at(key) += scaled_Jx;
      } else {
        subgraph_gradient[sub_map1].insert(key, scaled_Jx);
      }

      if (subgraph_gradient[sub_map2].exists(key)) {
        subgraph_gradient[sub_map2].at(key) += scaled_Jz;
      } else {
        subgraph_gradient[sub_map2].insert(key, scaled_Jz);
      }
    } else {
      // kind of dual residual
      POSE_TYPE pose_z_old = initial_old[sub_map2].at<POSE_TYPE>(key);
      pose_z = subinitials_[sub_map2].at<POSE_TYPE>(key);
      Vector bz = POSE_TYPE::Logmap(pose_z_old.between(pose_z));

      Values valuesForGradient;
      valuesForGradient.insert(0, pose_x);
      valuesForGradient.insert(1, pose_z);
      BetweenFactor<POSE_TYPE> factor = BetweenFactor<POSE_TYPE>(0, 1, POSE_TYPE(), rhoNoiseModel);
      d_res = d_res + factor.linearize(valuesForGradient)->gradientAtZero().norm();
      // d_res = d_res + rho * bz.norm(); // old version
    }

    // primal residual (Body11fnt, pag 18)
#if USE_L2_NORM == 0
    double bnorm = b.norm();
    if (bnorm > p_res) p_res = bnorm;
#else
    p_res = p_res + b.norm();
#endif

    // update of scaled dual variables, component-wise
    for (size_t k = 0; k < dualDimension_; k++) {
      uall_k[dualDimension_ * i + k] = uall_k[dualDimension_ * i + k] + b[k];
    }
  }

  if (computeSubgraphGradient_) {
    // Iterate over subgraphs and add \delta f(sub_id)
    for (int sub_id = 0; sub_id < num_subgraphs; sub_id++) {
      double gradient_norm = subgraph_gradient[sub_id].norm();
      d_res = d_res + gradient_norm * gradient_norm;
    }
    d_res = sqrt(d_res);
  }

  // return result of current iteration, and primal/dual residuals
  return std::make_pair(p_res, d_res);
}

/********************************************************************************************/
// ADMM - return sub_initials (optimized), and a std::vector of dual variables
template <class POSE_TYPE>
std::pair<std::vector<Values>, std::vector<double>> ADMM<POSE_TYPE>::optimize() {
  // Config
  double p_res, d_res;  // primal and dual residual, used to check stopping conditions

  // Residual log
  std::string residualLog = outputDir_ + "/residualLog.txt";
  residualLog_->open(residualLog.c_str(), std::fstream::out);

  if (verbosity_ >= DEBUG) {
    std::cout << "========================================" << std::endl;
    std::cout << "Parameters:  "
            "\n-rho: "
         << rho_ << "(initial)"
         << "\n-mu: " << mu_ << "\n-tau: " << tau_ << "\n-maxIter: " << maxIter_ << "\n-isParallel: " << isParallel_
         << "\n-adaptivePenalty: " << adaptivePenalty_ << "\n-min_p_res_: " << min_p_res_
         << "\n-min_d_res_: " << min_d_res_ << "\n-num_subgraphs: " << subgraphs_.size()
         << "\n-solver-type: " << solver_ << std::endl;
  }

  logCount_ = 0, logCountThresh_ = 10;

  // Start ADMM Timer
  ADMM<POSE_TYPE>::Time admm_start = boost::posix_time::microsec_clock::local_time();

  // Flagged initialization
  if (useFlaggedInitialization_) FlaggedInitialization();

  // run ADMM iterations
  for (int iter = 0; iter < maxIter_; iter++) {
    // Start iter timer
    tic();

    if (verbosity_ >= DEBUG) std::cout << "============== Iter: " << iter << " ==============" << std::endl;

    // run single ADMM iteration
    boost::tie(p_res, d_res) = iterate(iter);
    std::cout << "iter: " << iter << " : " << p_res << " | " << d_res << std::endl;

    // Stop timer
    double time_duration = toc();

    // update penalty parameter, when using adaptivePenalty
    if (adaptivePenalty_ && p_res > mu_ * d_res)
      rho_ = rho_ * tau_;  // increase penalty, to try to reduce primal infeasibility
    if (adaptivePenalty_ && d_res > mu_ * p_res) rho_ = rho_ / tau_;  // reduce penalty

    // log the graph after every iteration
    if (verbosity_ >= LOG) {
      logResult(iter, p_res, d_res, time_duration);
    }

    logCount_++;

    // print debug info
    if (verbosity_ >= DEBUG) {
      printf("p_res(iter) : %g (%d) < %g: min_p_res_ \n", p_res, iter, min_p_res_);
      printf("d_res(iter) : %g (%d) < %g: min_d_res_ \n", d_res, iter, min_d_res_);
      std::cout << "rho: " << rho_ << " (tau=" << tau_ << ", mu=" << mu_ << ")" << std::endl;
      if (iter == maxIter_ - 1) std::cout << "Reached maxIter: " << maxIter_ - 1 << std::endl;
    }

    // check stopping conditions
    if (p_res < min_p_res_ && d_res < min_d_res_) {
      if (verbosity_ >= DEBUG) {
        std::cout << "========================================" << std::endl;
        printf("p_res(iter) : %g (%d) < %g: min_p_res_ \n", p_res, iter, min_p_res_);
        printf("d_res(iter) : %g (%d) < %g: min_d_res_ \n", d_res, iter, min_d_res_);
      }

      // Break if stopping condition has reached
      break;
    }
  }

  if (verbosity_ >= DEBUG) {
    // Stop ADMM Timer
    ADMM<POSE_TYPE>::Time admm_end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration time_diff = admm_end - admm_start;
    std::cout << "Total time: " << time_diff.total_microseconds() << std::endl;
  }

  residualLog_->close();

  // unscale dual variables
  std::vector<double> y_k(uall_k.size(), 0.0f);
  for (size_t i = 0; i < y_k.size(); i++) y_k[i] = uall_k[i] * rho_;  // unscaled dual variables

  // return ADMM primal and dual solution
  return std::make_pair(subinitials_, y_k);
}
