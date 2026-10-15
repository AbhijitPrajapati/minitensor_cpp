#pragma once

#include "nn/gradients.hpp"
#include "nn/leafification.hpp"
#include "nn/loss.hpp"
#include "nn/modules/linear.hpp"
#include "nn/optimizers/sgd.hpp"
#include "nn/parameter.hpp"
#include "nn/parameter_tree.hpp"

// Parameter trees are seperate from composed module heirachies for flexibility. Optimizer state is 
// also seperate from the constant optimizer.