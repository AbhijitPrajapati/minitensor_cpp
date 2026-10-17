#pragma once

namespace minitensor::detail
{
	class VjpRegistry;

	void register_elementwise_vjps(VjpRegistry& registry);
	void register_linalg_vjps(VjpRegistry& registry);
	void register_manipulation_vjps(VjpRegistry& registry);
	void register_reduction_vjps(VjpRegistry& registry);
}
