use nalgebra as na; 

/// slam book eq. 2.4 
fn skew(v: na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// slam book eq. 3.27
fn calculate_j(theta: f64, n: &na::Vector3<f64>) -> na::Matrix3<f64> {
    let i3 = na::Matrix3::<f64>::identity(); 
    // need to check value of theta 
    // otherwise this function will be super slow 
    if theta > 1e-6 {
        (theta.sin() / theta) * i3 
        + (1.0 - theta.sin() / theta) * n * n.transpose() 
        + (1.0 - theta.cos()) / theta * skew(*n)
    } else {
        i3 
    }
}

/// slam book p. 65 
fn exp_map(v: &na::Vector6<f64>) -> na::Isometry3<f64> {
    let rho = v.fixed_slice::<3, 1>(0,0); 
    let phi = v.fixed_slice::<3, 1>(3,0); 
    
    let theta = phi.norm();
    let n = phi / theta; 
    let exp_phi_skew = theta.cos() * na::Matrix3::<f64>::identity() 
        + (1.0 - theta.cos()) * n * n.transpose() 
        + theta.sin() * skew(n);  

    let j = calculate_j(theta, &n);   

    let t = na::Translation3::from(j * rho); 
    let r = na::Rotation3::from_matrix(&exp_phi_skew); 
    na::Isometry3::from_parts(t, r.into())
}

/// Converts SE3 to se3
/// slam book p. 65 
fn log_map(v: &na::Isometry3<f64>) -> na::Vector6<f64> {
    let t: na::Vector3<f64> = v.translation.vector;
    let quat = v.rotation; 

    // quaternion to angle-axis 
    // slam book eq. 2.44 
    let theta: f64 = 2.0 * (quat.scalar()).acos();
    let n = quat.vector() * theta / ((0.5 * theta).sin() + 1e-6);
    let phi = theta * n; 

    let j = calculate_j(theta, &n);
    let j_inv = j.try_inverse().unwrap(); 
    let rho = j_inv * t; 

    let mut ret = na::Vector6::<f64>::zeros(); 
    ret.fixed_slice_mut::<3,1>(0, 0).copy_from(&rho); 
    ret.fixed_slice_mut::<3,1>(3, 0).copy_from(&phi); 

    ret 
}

/// Jacobian of exponential map 
/// jacob. of transformed point wrt se3 
/// ref slam book p. 70 
fn exp_map_jacobian(transformed_point: &na::Point3<f64>) -> na::Matrix3x6<f64> {
    let mut ss = na::Matrix3x6::zeros(); 
    ss.fixed_slice_mut::<3,3>(0,0)
        .copy_from(&na::Matrix3::<f64>::identity()); 
    ss.fixed_slice_mut::<3,3>(0, 3)
        .copy_from(&(-skew(transformed_point.coords)));
    ss 
}

/// Projects a point in camera frame to image 
fn project(
    // fx, fy, cx, cy
    params: &na::Vector4<f64>, 
    pt: &na::Point3<f64>, 
) -> na::Point2<f64> {
    na::Point2::<f64>::new(
        // fx*x / z + cx
        params[0] * pt.x / pt.z + params[2],
        params[1] * pt.y / pt.z + params[3],  
    )
}

/// Jacobian of projection wrt fx, fy, cx, cy 
fn proj_jacobian_wrt_params(transformed_pt: &na::Point3<f64>) -> na::Matrix2x4<f64> {
    na::Matrix2x4::<f64>::new(
        transformed_pt.x / transformed_pt.z, 
        0.0, 
        1.0, 
        0.0, 
        0.0,
        transformed_pt.y / transformed_pt.z, 
        0.0, 
        1.0, 
    )
}

/// Jacobian of projection wrt 3D point in camera frame 
/// ref slam book eq. 6.43
fn proj_jacobian_wrt_point(
    // fx, fy, cx, cy
    camera_model: &na::Vector4<f64>,
    transformed_pt: &na::Point3<f64>,  
) -> na::Matrix2x3<f64> {
    na::Matrix2x3::<f64>::new(
        camera_model[0]/transformed_pt.z, 
        0.0, 
        -transformed_pt.x*camera_model[0]/transformed_pt.z.powi(2), 
        0.0,
        camera_model[1]/transformed_pt.z, 
        -transformed_pt.y*camera_model[1]/transformed_pt.z.powi(2), 
    )
}

struct Calibration<'a> {
    model_pts: &'a Vec<na::Point3<f64>>,
    image_pts_set: &'a Vec<Vec<na::Point2<f64>>>, 
}

impl<'a> Calibration<'a> {
    /// State: fx, fy, cx, cy, se3 = 4 + 6 dim * num of images 
    fn decode_params(
        &self, 
        param: &na::DVector<f64>, 
    ) -> (na::Vector4<f64>, Vec<na::Isometry3<f64>>) {
        let camera_model: na::Vector4<f64> = param.fixed_slice::<4,1>(0,0).clone_owned(); 

        let transforms = self.
            image_pts_set
            .iter()
            .enumerate()
            .map(
                |(i, _)| 
                {
                    let se3: na::Vector6<f64> = param.fixed_slice::<6,1>(4+6*i, 0).clone_owned(); 
                    exp_map(&se3)
                }
            ).collect::<Vec<_>>(); 
        
        (camera_model, transforms)
    }
}

/// Must implement apply function for residual 
/// and jacobian function 
impl Calibration<'_> {
    /// Calculate residual 
    fn apply(&self, p: &na::DVector<f64>) -> na::DVector<f64> {
        let (camera_model, transforms) = self.decode_params(p); 

        let num_images = self.image_pts_set.len(); 
        let num_target_points = self.model_pts.len(); 
        let num_residuals = num_images * num_target_points; 

        let mut residual = na::DVector::<f64>::zeros(num_residuals * 2); 

        let mut residual_idx = 0; 
        for (image_pts, transform) in self.image_pts_set.iter().zip(transforms.iter()) {
            for (observed_image_pt, target_pt) in image_pts.iter().zip(self.model_pts.iter()) {
                // transform the point from model frame to camera frame  
                let transformed_point = transform * target_pt; 
                // project the point to image 
                let projected_pt = project(&camera_model, &transformed_point);
                
                let individual_residual = projected_pt - observed_image_pt; 
                residual
                    .fixed_slice_mut::<2,1>(residual_idx, 0)
                    .copy_from(&individual_residual);
                residual_idx += 2; 
            }
        }

        residual
    }

    fn jacobian(&self, p: &na::DVector<f64>) -> na::DMatrix<f64> {
        let (camera_model, transforms) = self.decode_params(p); 

        let num_images = self.image_pts_set.len();
        let num_target_points = self.model_pts.len(); 
        let num_residuals = num_images * num_target_points; 
        let num_unknowns = 6*num_images + 4; 

        let mut jacobian = na::DMatrix::<f64>::zeros(num_residuals*2, num_unknowns); 

        let mut residual_idx = 0; 
        for (tfrom_idx, transform) in transforms.iter().enumerate() {
            for target_pt in self.model_pts.iter() {
                let transformed_point = transform * target_pt; 

                jacobian
                    .fixed_slice_mut::<2,4>(residual_idx, 0)
                    .copy_from(&proj_jacobian_wrt_params(&transformed_point)); 

                let proj_jacobian_wrt_point = 
                    proj_jacobian_wrt_point(&camera_model, &transformed_point);
                let transform_jacobian_wrt_transform = exp_map_jacobian(&transformed_point); 
            
                jacobian
                    .fixed_slice_mut::<2, 6>(residual_idx, 4+tfrom_idx*6)
                    .copy_from(&(proj_jacobian_wrt_point*transform_jacobian_wrt_transform)); 

                residual_idx += 2; 
            }
        }

        jacobian
    }

    /// slam book sec. 5.2.2
    fn gauss_newton(&self, params: &na::DVector<f64>, max_iter: usize, tolerance: f64) -> na::DVector<f64> {
        // params size is mx1 
        let mut params = params.clone(); 

        for _ in 0..max_iter {
            // residual size is 2n x 1 
            let residual = self.apply(&params);
            // jacobian size is 2n x m 
            let jacobian = self.jacobian(&params);

            // Solve the normal equations: J^T * J * delta_params = J^T * residual
            // svd solve Solves the system self * x = b where self is the decomposed matrix and x the unknown.
            let delta_params = na::linalg::SVD::new(&jacobian.transpose() * &jacobian, true, true)
                .solve(&(&jacobian.transpose() * &residual), tolerance)
                .unwrap_or(na::DVector::zeros(params.len()));

            params -= delta_params.clone();

            // Check for convergence
            if delta_params.norm() < tolerance {
                break;
            }
        }

        params
    }
}

fn main() {
    // 11x11 pattern in a 1m x 1m square in the XY plane
    let mut source_pts: Vec<na::Point3<f64>> = Vec::new();
    for i in -5..6 {
        for j in -5..6 {
            source_pts.push(na::Point3::<f64>::new(i as f64 * 0.1, j as f64 * 0.1, 0.0));
        }
    }

    // Ground truth camera model
    let camera_model = na::Vector4::<f64>::new(540.0, 540.0, 320.0, 240.0); // fx, fy, cx, cy

    // Ground truth camera-from-model transforms for three "images"
    // new: translation, axisangle
    let transforms = vec![
        na::Isometry3::<f64>::new(
            na::Vector3::<f64>::new(-0.1, 0.1, 2.0),
            na::Vector3::<f64>::new(-0.2, 0.2, 0.2),
        ),
        na::Isometry3::<f64>::new(
            na::Vector3::<f64>::new(-0.1, -0.1, 2.0),
            na::Vector3::<f64>::new(0.2, -0.2, 0.2),
        ),
        na::Isometry3::<f64>::new(
            na::Vector3::<f64>::new(0.1, 0.1, 2.0),
            na::Vector3::<f64>::new(-0.2, -0.2, -0.2),
        ),
    ];

    // transform from model frame to camera frame 
    let transformed_pts = transforms
        .iter()
        .map(|t| source_pts.iter().map(|p| t*p).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    // project to image 
    let imaged_pts = transformed_pts
        .iter()
        .map(|t_list| {
            t_list
                .iter()
                .map(|t| project(&camera_model, t))
                .collect::<Vec<na::Point2<f64>>>()
        })
        .collect::<Vec<_>>(); 

    // create calibration data 
    let calibration_solver = Calibration {
        model_pts: &source_pts, 
        image_pts_set: &imaged_pts, 
    };

    // initial guess 
    let mut init_param = na::DVector::<f64>::zeros(4+imaged_pts.len()*6); 

    // Arbitrary guess for camera model
    // NOTE, cannot be too far away from gt 
    init_param[0] = 510.0; // fx
    init_param[1] = 510.0; // fy
    init_param[2] = 300.0; // cx
    init_param[3] = 200.0; // cy

    // Arbitrary guess for poses (3m in front of the camera with no rotation)
    // We have to convert this to a 6D lie algebra element to populate the parameter
    // vector.
    let init_pose_lie = log_map(&na::Isometry3::translation(0.0, 0.0, 3.0));

    init_param
        .fixed_slice_mut::<6, 1>(4, 0)
        .copy_from(&init_pose_lie);
    init_param
        .fixed_slice_mut::<6, 1>(4 + 6, 0)
        .copy_from(&init_pose_lie);
    init_param
        .fixed_slice_mut::<6, 1>(4 + 6 * 2, 0)
        .copy_from(&init_pose_lie);

    // Solve with Gauss Newton
    let max_iter = 100;
    let tolerance = 1e-6;
    let res: na::Matrix<f64, na::Dynamic, na::Const<1>, na::VecStorage<f64, na::Dynamic, na::Const<1>>> = calibration_solver.gauss_newton(&init_param, max_iter, tolerance);

    eprintln!("{}\n\n", res);

    // Print intrinsics results
    eprintln!("ground truth intrinsics: {}", camera_model);
    eprintln!(
        "optimized intrinsics: {}",
        res.fixed_slice::<4, 1>(0, 0)
    );

    // Print transforms
    for (i, t) in transforms.iter().enumerate() {
        eprintln!("ground truth transform[{}]: {}", i, t);
        eprintln!(
            "optimized result[{}]: {}\n",
            i,
            exp_map(
                &res.fixed_slice::<6, 1>(4 + 6 * i, 0)
                    .clone_owned()
            )
        );
    }
}

