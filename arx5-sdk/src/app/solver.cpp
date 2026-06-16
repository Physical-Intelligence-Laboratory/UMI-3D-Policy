#include "app/solver.h"

#include <random>
#include <sstream>
#include <stdexcept>

namespace arx
{

KDL::Frame Arx5Solver::pose6d_to_kdl(const Eigen::Matrix<double, 6, 1> &pose)
{
    const double x = pose(0);
    const double y = pose(1);
    const double z = pose(2);
    const double roll = pose(3);
    const double pitch = pose(4);
    const double yaw = pose(5);

    return KDL::Frame(KDL::Rotation::RPY(roll, pitch, yaw), KDL::Vector(x, y, z));
}

Eigen::Matrix<double, 6, 1> Arx5Solver::kdl_to_pose6d(const KDL::Frame &frame)
{
    Eigen::Matrix<double, 6, 1> pose;
    double roll, pitch, yaw;
    frame.M.GetRPY(roll, pitch, yaw);
    pose << frame.p.x(), frame.p.y(), frame.p.z(), roll, pitch, yaw;
    return pose;
}

void Arx5Solver::validate_joint_vector_size(const Eigen::VectorXd &vec, const std::string &name) const
{
    if (vec.size() != JOINT_DOF_)
    {
        std::ostringstream oss;
        oss << "Arx5Solver::" << name << " size mismatch, expected " << JOINT_DOF_ << ", got " << vec.size();
        throw std::runtime_error(oss.str());
    }
}

bool Arx5Solver::within_joint_limit(const Eigen::VectorXd &q, double tol) const
{
    if (q.size() != JOINT_DOF_)
    {
        return false;
    }

    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        if (q(i) < JOINT_POS_MIN_(i) - tol || q(i) > JOINT_POS_MAX_(i) + tol)
        {
            return false;
        }
    }
    return true;
}

Arx5Solver::Arx5Solver(std::string urdf_path, int joint_dof, Eigen::VectorXd joint_pos_min,
                       Eigen::VectorXd joint_pos_max)
    : Arx5Solver(urdf_path, joint_dof, joint_pos_min, joint_pos_max, "base_link", "eef_link",
                 (Eigen::Vector3d() << 0.0, 0.0, -9.807).finished())
{
}

Arx5Solver::Arx5Solver(std::string urdf_path, int joint_dof, Eigen::VectorXd joint_pos_min,
                       Eigen::VectorXd joint_pos_max, std::string base_link, std::string eef_link,
                       Eigen::Vector3d gravity_vector)
    : JOINT_DOF_(joint_dof), JOINT_POS_MIN_(joint_pos_min), JOINT_POS_MAX_(joint_pos_max)
{
    if (JOINT_DOF_ <= 0)
    {
        throw std::runtime_error("Arx5Solver: joint_dof must be > 0");
    }

    if (JOINT_POS_MIN_.size() != JOINT_DOF_ || JOINT_POS_MAX_.size() != JOINT_DOF_)
    {
        throw std::runtime_error("Arx5Solver: joint_pos_min/max size does not equal joint_dof");
    }

    if (!kdl_parser::treeFromFile(urdf_path, tree_))
    {
        throw std::runtime_error("Arx5Solver: Failed to build KDL tree from URDF: " + urdf_path);
    }

    if (!tree_.getChain(base_link, eef_link, chain_))
    {
        throw std::runtime_error("Arx5Solver: Failed to get chain from kdl tree");
    }

    chain_without_fixed_joints_ = KDL::Chain();
    for (unsigned int i = 0; i < chain_.getNrOfSegments(); ++i)
    {
        const KDL::Segment &seg = chain_.getSegment(i);
        if (seg.getJoint().getType() != KDL::Joint::None)
        {
            chain_without_fixed_joints_.addSegment(seg);
        }
    }

    if (static_cast<int>(chain_without_fixed_joints_.getNrOfJoints()) != JOINT_DOF_)
    {
        std::ostringstream oss;
        oss << "Arx5Solver: Parsed " << chain_without_fixed_joints_.getNrOfJoints()
            << " non-fixed joints, which does not equal to joint_dof = " << JOINT_DOF_;
        throw std::runtime_error(oss.str());
    }

    // FK should reach the real eef_link
    fk_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain_);

    // IK and ID should only use active joints
    ik_solver_ = std::make_shared<KDL::ChainIkSolverPos_LMA>(chain_without_fixed_joints_, EPS_, MAXITER_, EPS_JOINTS_);

    id_solver_ = std::make_shared<KDL::ChainIdSolver_RNE>(
        chain_without_fixed_joints_, KDL::Vector(gravity_vector(0), gravity_vector(1), gravity_vector(2)));
}

Eigen::VectorXd Arx5Solver::inverse_dynamics(Eigen::VectorXd joint_pos, Eigen::VectorXd joint_vel,
                                             Eigen::VectorXd joint_acc)
{
    validate_joint_vector_size(joint_pos, "inverse_dynamics(joint_pos)");
    validate_joint_vector_size(joint_vel, "inverse_dynamics(joint_vel)");
    validate_joint_vector_size(joint_acc, "inverse_dynamics(joint_acc)");

    KDL::JntArray q(JOINT_DOF_);
    KDL::JntArray qd(JOINT_DOF_);
    KDL::JntArray qdd(JOINT_DOF_);
    KDL::JntArray tau(JOINT_DOF_);

    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        q(i) = joint_pos(i);
        qd(i) = joint_vel(i);
        qdd(i) = joint_acc(i);
    }

    std::vector<KDL::Wrench> f_ext(chain_without_fixed_joints_.getNrOfSegments(), KDL::Wrench::Zero());

    const int ret = id_solver_->CartToJnt(q, qd, qdd, f_ext, tau);
    if (ret < 0)
    {
        std::ostringstream oss;
        oss << "Arx5Solver::inverse_dynamics failed with KDL status " << ret;
        throw std::runtime_error(oss.str());
    }

    Eigen::VectorXd out(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        out(i) = tau(i);
    }
    return out;
}

std::tuple<int, Eigen::VectorXd> Arx5Solver::inverse_kinematics(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                                Eigen::VectorXd current_joint_pos)
{
    if (current_joint_pos.size() != JOINT_DOF_)
    {
        return std::make_tuple(KDL::SolverI::E_SIZE_MISMATCH, Eigen::VectorXd::Zero(JOINT_DOF_));
    }

    const KDL::Frame target_frame = pose6d_to_kdl(target_pose_6d);

    KDL::JntArray q_init(JOINT_DOF_);
    KDL::JntArray q_result(JOINT_DOF_);

    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        q_init(i) = current_joint_pos(i);
    }

    const int ik_status = ik_solver_->CartToJnt(q_init, target_frame, q_result);

    Eigen::VectorXd q_out(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        q_out(i) = q_result(i);
    }

    if (ik_status != KDL::SolverI::E_NOERROR)
    {
        return std::make_tuple(ik_status, q_out);
    }

    if (!within_joint_limit(q_out))
    {
        return std::make_tuple(E_EXCEED_JOITN_LIMIT, q_out);
    }

    return std::make_tuple(KDL::SolverI::E_NOERROR, q_out);
}

std::tuple<int, Eigen::VectorXd> Arx5Solver::multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                                            Eigen::VectorXd current_joint_pos, int additional_trial_num)
{
    if (additional_trial_num < 0)
    {
        throw std::invalid_argument("Arx5Solver::multi_trial_ik additional_trial_num must be non-negative");
    }

    if (current_joint_pos.size() != JOINT_DOF_)
    {
        return std::make_tuple(KDL::SolverI::E_SIZE_MISMATCH, Eigen::VectorXd::Zero(JOINT_DOF_));
    }

    // First try current joint position
    {
        std::tuple<int, Eigen::VectorXd> ret = inverse_kinematics(target_pose_6d, current_joint_pos);
        if (std::get<0>(ret) == KDL::SolverI::E_NOERROR)
        {
            return ret;
        }
    }

    // Then try zero pose
    {
        Eigen::VectorXd zero_seed = Eigen::VectorXd::Zero(JOINT_DOF_);
        std::tuple<int, Eigen::VectorXd> ret = inverse_kinematics(target_pose_6d, zero_seed);
        if (std::get<0>(ret) == KDL::SolverI::E_NOERROR)
        {
            return ret;
        }
    }

    // Then random seeds within joint limits
    std::random_device rd;
    std::mt19937 gen(rd());

    int best_status = KDL::SolverI::E_NO_CONVERGE;
    Eigen::VectorXd best_q = current_joint_pos;

    for (int trial = 0; trial < additional_trial_num; ++trial)
    {
        Eigen::VectorXd q_seed(JOINT_DOF_);
        for (int i = 0; i < JOINT_DOF_; ++i)
        {
            std::uniform_real_distribution<double> dist(JOINT_POS_MIN_(i), JOINT_POS_MAX_(i));
            q_seed(i) = dist(gen);
        }

        std::tuple<int, Eigen::VectorXd> ret = inverse_kinematics(target_pose_6d, q_seed);
        int status = std::get<0>(ret);
        const Eigen::VectorXd &q = std::get<1>(ret);

        if (status == KDL::SolverI::E_NOERROR)
        {
            return ret;
        }

        best_status = status;
        best_q = q;
    }

    return std::make_tuple(best_status, best_q);
}

std::string Arx5Solver::get_ik_status_name(int ik_status)
{
    auto it = IK_STATUS_MAP_.find(ik_status);
    if (it != IK_STATUS_MAP_.end())
    {
        return it->second;
    }

    return "UNKNOWN_IK_STATUS_" + std::to_string(ik_status);
}

Eigen::Matrix<double, 6, 1> Arx5Solver::forward_kinematics(Eigen::VectorXd joint_pos)
{
    validate_joint_vector_size(joint_pos, "forward_kinematics");

    KDL::JntArray q(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
    {
        q(i) = joint_pos(i);
    }

    KDL::Frame frame_out;
    const int ret = fk_solver_->JntToCart(q, frame_out);
    if (ret < 0)
    {
        std::ostringstream oss;
        oss << "Arx5Solver::forward_kinematics failed with KDL status " << ret;
        throw std::runtime_error(oss.str());
    }

    return kdl_to_pose6d(frame_out);
}

} // namespace arx