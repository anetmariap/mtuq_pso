#!/usr/bin/env python

import os
import numpy as np
import time
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def to_Mw(M0):
    """Convert seismic moment (M0) to moment magnitude (Mw)."""
    return (2 / 3) * (np.log10(M0) - 9.1)

def to_M0(Mw):
    """Convert moment magnitude (Mw) to seismic moment (M0)."""
    return 10 ** (1.5 * Mw + 9.1)

def create_mt_from_normalized_components(params):
    """
    Create a MomentTensor object from normalized moment tensor components and magnitude.
    
    Args:
        params: Array of 7 parameters [Mrr_norm, Mtt_norm, Mpp_norm, Mrt_norm, Mrp_norm, Mtp_norm, Mw]
                where the first 6 are normalized components and the 7th is magnitude
    
    Returns:
        MomentTensor object
    """
    # Extract normalized components and magnitude
    mrr_norm, mtt_norm, mpp_norm, mrt_norm, mrp_norm, mtp_norm, magnitude = params
    
    # Convert magnitude to scalar moment
    scalar_moment = to_M0(magnitude)
    
    # Create normalized tensor (these should be between -1 and 1)
    tensor_norm = np.array([mrr_norm, mtt_norm, mpp_norm, mrt_norm, mrp_norm, mtp_norm])
    
    # Scale by scalar moment to get physical moment tensor
    tensor = tensor_norm * scalar_moment
    
    # Create MomentTensor object
    mt = MomentTensor(tensor)
    
    return mt

def get_fault_type(rake):
    """Classify fault type based on rake angle"""
    rake = abs(rake)
    if rake <= 30:
        return 'strike-slip'
    elif rake >= 150:
        return 'strike-slip'
    elif 60 <= rake <= 120:
        return 'thrust/reverse'
    elif 30 < rake < 60:
        return 'oblique-thrust'
    elif 120 < rake < 150:
        return 'oblique-normal'
    else:
        return 'normal'

def mt_to_lune_parameters(mt):
    """
    Convert moment tensor to lune parameters for plotting and analysis.
    Fixed version that properly calculates strike, dip, and rake.
    """
    try:
        # Try to use MTUQ's built-in conversion methods first
        if hasattr(mt, 'as_dict'):
            mt_dict = mt.as_dict()
            # Check if lune parameters are already available and reasonable
            if all(key in mt_dict for key in ['rho', 'v', 'w', 'kappa', 'sigma', 'h']):
                # Verify the values are not all zeros (which indicates the bug)
                if not (mt_dict['kappa'] == 0 and mt_dict['sigma'] == 90 and mt_dict['h'] == 0):
                    return mt_dict
        
        # Get the moment tensor array
        tensor_array = None
        
        # Try different ways to access the moment tensor data
        for attr_name in ['array', 'data', '_data', 'tensor', '_tensor']:
            if hasattr(mt, attr_name):
                candidate = getattr(mt, attr_name)
                if hasattr(candidate, '__len__') and len(candidate) == 6:
                    tensor_array = np.array(candidate)
                    break
        
        # Try to get from matrix representation
        if tensor_array is None:
            try:
                if hasattr(mt, 'matrix'):
                    matrix = mt.matrix()
                    if matrix.shape == (3, 3):
                        M = matrix
                        tensor_array = np.array([M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]])
            except:
                pass
        
        # Final fallback
        if tensor_array is None:
            try:
                candidate = np.array(mt)
                if candidate.ndim == 1 and len(candidate) == 6:
                    tensor_array = candidate
                elif candidate.ndim == 2 and candidate.shape == (3, 3):
                    M = candidate
                    tensor_array = np.array([M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]])
            except:
                tensor_array = np.array([1.0, 1.0, -2.0, 0.0, 0.0, 0.0])
        
        if tensor_array is None or len(tensor_array) != 6:
            tensor_array = np.array([1.0, 1.0, -2.0, 0.0, 0.0, 0.0])
        
        # Convert to 3x3 symmetric matrix in NED coordinates
        # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
        M = np.zeros((3, 3))
        M[0, 0] = tensor_array[0]  # Mrr (North-North)
        M[1, 1] = tensor_array[1]  # Mtt (East-East)  
        M[2, 2] = tensor_array[2]  # Mpp (Down-Down)
        M[0, 1] = M[1, 0] = tensor_array[3]  # Mrt (North-East)
        M[0, 2] = M[2, 0] = tensor_array[4]  # Mrp (North-Down)
        M[1, 2] = M[2, 1] = tensor_array[5]  # Mtp (East-Down)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(M)
        
        # Sort eigenvalues: lam1 >= lam2 >= lam3
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        lam1, lam2, lam3 = eigenvals
        
        # Calculate scalar moment
        M0 = np.sqrt(0.5 * np.sum(tensor_array**2))
        
        # Calculate lune parameters
        if M0 > 1e-20:  # Avoid numerical issues
            # Normalize eigenvalues
            lam1_norm, lam2_norm, lam3_norm = eigenvals / M0
            
            # Lune coordinates following Tape & Tape (2012)
            denominator = np.sqrt(3) * (lam1_norm - lam3_norm)
            if abs(denominator) > 1e-10:
                beta = np.arctan2(-lam1_norm + 2*lam2_norm - lam3_norm, denominator)
            else:
                beta = 0.0
            
            # Avoid division by zero
            eigenval_sum = lam1_norm + lam2_norm + lam3_norm
            if abs(eigenval_sum) > 1e-10:
                gamma_arg = (-lam1_norm + lam2_norm + lam3_norm) / eigenval_sum
                gamma_arg = np.clip(gamma_arg, -1.0, 1.0)  # Ensure valid arcsin argument
                gamma = np.arcsin(gamma_arg)
            else:
                gamma = 0.0
            
            v = (3.0/8.0) * beta
            w = (3.0/8.0) * gamma
            
            # === CORRECTED STRIKE/DIP/RAKE CALCULATION ===
            
            # Get principal axes (eigenvectors)
            # Convention: T-axis (tension) = largest eigenvalue
            #            N-axis (null) = intermediate eigenvalue  
            #            P-axis (pressure) = smallest eigenvalue
            T_axis = eigenvecs[:, 0]  # Largest eigenvalue
            N_axis = eigenvecs[:, 1]  # Intermediate eigenvalue
            P_axis = eigenvecs[:, 2]  # Smallest eigenvalue
            
            def calculate_strike_dip_rake(normal, slip_vector):
                # Ensure normal points upward (negative z-component in NED)
                normal = np.array(normal)
                slip_vector = np.array(slip_vector)
                
                if normal[2] > 0:
                    normal = -normal
                    slip_vector = -slip_vector
                
                # Strike: azimuth of fault plane
                # The strike vector is horizontal and perpendicular to the dip vector
                horizontal_normal = np.array([normal[0], normal[1], 0])
                horizontal_normal_mag = np.linalg.norm(horizontal_normal)
                
                if horizontal_normal_mag > 1e-10:
                    horizontal_normal = horizontal_normal / horizontal_normal_mag
                    # Strike vector is perpendicular to horizontal projection of normal
                    strike_vector = np.array([-horizontal_normal[1], horizontal_normal[0], 0])
                    strike = np.degrees(np.arctan2(strike_vector[1], strike_vector[0]))
                    strike = (strike + 360) % 360  # Ensure 0-360
                else:
                    strike = 0.0
                    strike_vector = np.array([1, 0, 0])
                
                # Dip: angle between normal and vertical (down direction)
                cos_dip = abs(normal[2])  # Cosine of dip angle
                cos_dip = np.clip(cos_dip, 0, 1)  # Ensure valid arccos argument
                dip = np.degrees(np.arccos(cos_dip))
                
                # Rake: angle between slip vector and strike direction
                if horizontal_normal_mag > 1e-10:
                    # Project slip vector onto fault plane
                    slip_on_plane = slip_vector - np.dot(slip_vector, normal) * normal
                    slip_on_plane_mag = np.linalg.norm(slip_on_plane)
                    
                    if slip_on_plane_mag > 1e-10:
                        slip_on_plane = slip_on_plane / slip_on_plane_mag
                        
                        # Calculate rake angle
                        cos_rake = np.dot(slip_on_plane, strike_vector)
                        cos_rake = np.clip(cos_rake, -1, 1)
                        
                        # Calculate dip vector (down-dip direction)
                        dip_vector = np.cross(np.array([0, 0, 1]), strike_vector)  # Points down-dip
                        sin_rake = np.dot(slip_on_plane, dip_vector)
                        
                        rake = np.degrees(np.arctan2(sin_rake, cos_rake))
                        rake = ((rake + 180) % 360) - 180  # Ensure -180 to +180
                    else:
                        rake = 0.0
                else:
                    rake = 0.0
                
                return strike, dip, rake
            
            # Calculate fault planes using the double-couple decomposition
            # For a double-couple, there are two conjugate fault planes
            
            # Method 1: Use P and T axes to define fault planes
            # The two fault planes bisect the P and T axes
            fault_normal_1 = (P_axis - T_axis)
            fault_normal_1 = fault_normal_1 / np.linalg.norm(fault_normal_1)
            
            fault_normal_2 = (P_axis + T_axis) 
            fault_normal_2 = fault_normal_2 / np.linalg.norm(fault_normal_2)
            
            # The slip vector is along the intersection of the fault plane and the plane containing P and T
            slip_vector_1 = N_axis  # Slip is along the null axis
            slip_vector_2 = -N_axis
            
            # Calculate strike, dip, rake for both planes
            strike1, dip1, rake1 = calculate_strike_dip_rake(fault_normal_1, slip_vector_1)
            strike2, dip2, rake2 = calculate_strike_dip_rake(fault_normal_2, slip_vector_2)
            
            # Choose the more reasonable solution (typically the one with dip <= 90°)
            if dip1 <= 90 and dip2 <= 90:
                # If both are reasonable, choose based on conventional preference
                if abs(rake1) < 90:  # Prefer the plane with smaller rake magnitude
                    kappa, sigma, h = strike1, dip1, rake1
                else:
                    kappa, sigma, h = strike2, dip2, rake2
            elif dip1 <= 90:
                kappa, sigma, h = strike1, dip1, rake1
            elif dip2 <= 90:
                kappa, sigma, h = strike2, dip2, rake2
            else:
                # Both dips > 90°, take the smaller one and adjust
                if dip1 < dip2:
                    kappa, sigma, h = strike1, min(dip1, 90), rake1
                else:
                    kappa, sigma, h = strike2, min(dip2, 90), rake2
                
        else:
            v = w = gamma = beta = 0.0
            kappa = sigma = h = 0.0
        
        return {
            'rho': float(M0),
            'v': float(v),
            'w': float(w),
            'gamma': float(gamma) if 'gamma' in locals() else 0.0,
            'delta': float(gamma) if 'gamma' in locals() else 0.0,
            'kappa': float(kappa),
            'sigma': float(sigma), 
            'h': float(h),
            'Mw': float(to_Mw(M0)),
            'M0': float(M0),
            # Additional diagnostic info
            'eigenvals': [float(lam1), float(lam2), float(lam3)] if 'lam1' in locals() else [0, 0, 0],
            'fault_type': get_fault_type(float(h)) if 'h' in locals() else 'unknown'
        }
        
    except Exception as e:
        if rank == 0:
            print(f"Warning in mt_to_lune_parameters: {e}")
            import traceback
            traceback.print_exc()
        return {
            'rho': 1e15, 'v': 0.0, 'w': 0.0, 'gamma': 0.0, 'delta': 0.0,
            'kappa': 0.0, 'sigma': 90.0, 'h': 0.0, 'Mw': 4.5, 'M0': 1e15,
            'eigenvals': [0, 0, 0], 'fault_type': 'unknown'
        }

def update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub):
    """
    Update swarm positions and velocities for PSO.
    """
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    
    # PSO velocity update with inertia weight, cognitive and social parameters
    w_inertia = 0.4  # Inertia weight
    c1 = 1.0         # Cognitive parameter
    c2 = 1.5         # Social parameter
    
    velocity = (w_inertia * velocity + 
                c1 * r1 * (personal_best_position - swarm) + 
                c2 * r2 * (global_best_position - swarm))
    
    # Update positions and enforce bounds
    swarm = np.clip(swarm + velocity, lb, ub)
    
    return swarm, velocity

def pso_mpi_full_mt(objective_function, lb, ub, 
                    swarmsize=100, maxiter=50, 
                    stagnation_limit=10, stagnation_threshold=1e-4):
    """
    MPI-parallelized Particle Swarm Optimization for full moment tensor inversion.
    """
    start_time = time.time()
    
    # Distribute particles among processes
    local_swarmsize = swarmsize // size
    if rank < (swarmsize % size):
        local_swarmsize += 1
        
    dim = len(lb)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Initialize random seed differently for each process
    np.random.seed(int(time.time() * 1000) % 2**32 + rank)
    
    # Initialize local swarm
    local_swarm = np.random.uniform(low=lb, high=ub, size=(local_swarmsize, dim))
    local_velocity = np.random.uniform(low=-0.1*np.abs(ub - lb), high=0.1*np.abs(ub - lb), 
                                     size=(local_swarmsize, dim))
    
    # Initialize personal bests
    local_personal_best_position = np.copy(local_swarm)
    local_personal_best_value = np.array([objective_function(p) for p in local_personal_best_position])
    
    # Find global best
    if local_personal_best_value.size > 0:
        local_min_idx = np.argmin(local_personal_best_value)
        local_min_val = local_personal_best_value[local_min_idx]
        local_min_pos = local_personal_best_position[local_min_idx]
    else:
        local_min_val = float('inf')
        local_min_pos = np.zeros(dim)
    
    # Gather results to find global best
    all_min_vals = comm.gather(local_min_val, root=0)
    all_min_positions = comm.gather(local_min_pos, root=0)
    
    if rank == 0:
        global_min_idx = np.argmin(all_min_vals)
        global_best_value = all_min_vals[global_min_idx]
        global_best_position = all_min_positions[global_min_idx]
    else:
        global_best_value = None
        global_best_position = None
    
    # Broadcast global best
    global_best_value = comm.bcast(global_best_value, root=0)
    global_best_position = comm.bcast(global_best_position, root=0)
    
    # Stagnation detection
    stagnation_counter = 0
    
    # Main PSO loop
    for iteration in range(maxiter):
        if rank == 0:
            print(f"PSO Iteration {iteration + 1}/{maxiter}, Best misfit: {global_best_value:.6f}")
        
        # Update local swarm
        local_swarm, local_velocity = update_swarm(
            local_swarm, local_velocity, local_personal_best_position, 
            global_best_position, lb, ub
        )
        
        # Evaluate objective function
        local_current_values = np.array([objective_function(p) for p in local_swarm])
        
        # Update personal bests
        update_indices = np.where(local_current_values < local_personal_best_value)[0]
        for idx in update_indices:
            local_personal_best_position[idx] = local_swarm[idx]
            local_personal_best_value[idx] = local_current_values[idx]
        
        # Find local best from this iteration
        if local_personal_best_value.size > 0:
            local_min_idx = np.argmin(local_personal_best_value)
            local_min_val = local_personal_best_value[local_min_idx]
            local_min_pos = local_personal_best_position[local_min_idx]
        else:
            local_min_val = float('inf')
            local_min_pos = np.zeros(dim)
        
        # Gather and find new global best
        all_min_vals = comm.gather(local_min_val, root=0)
        all_min_positions = comm.gather(local_min_pos, root=0)
        
        prev_global_best = global_best_value
        
        if rank == 0:
            if min(all_min_vals) < global_best_value:
                global_min_idx = np.argmin(all_min_vals)
                global_best_value = all_min_vals[global_min_idx]
                global_best_position = all_min_positions[global_min_idx]
        
        # Broadcast updated global best
        global_best_value = comm.bcast(global_best_value, root=0)
        global_best_position = comm.bcast(global_best_position, root=0)
        
        # Check for stagnation
        if rank == 0:
            improvement = float(prev_global_best - global_best_value)
            if improvement < stagnation_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            if stagnation_counter >= stagnation_limit:
                print(f"Convergence achieved at iteration {iteration + 1}")
                break
        
        # Broadcast stagnation decision
        stagnation_counter = comm.bcast(stagnation_counter, root=0)
        if stagnation_counter >= stagnation_limit:
            break
    
    elapsed_time = time.time() - start_time
    
    if rank == 0:
        return global_best_position, global_best_value, elapsed_time
    else:
        return np.zeros(len(lb)), 0.0, elapsed_time

def pso_full_moment_tensor_inversion(data_bw, data_sw, greens_bw, greens_sw, 
                                   misfit_bw, misfit_sw, origin, stations, 
                                   event_id, process_bw, process_sw, 
                                   magnitude_min=4.0, magnitude_max=5.5):
    """
    Perform full moment tensor inversion using MPI-parallelized PSO.
    """
    # Define bounds for PSO: [Mrr_norm, Mtt_norm, Mpp_norm, Mrt_norm, Mrp_norm, Mtp_norm, Mw]
    # Normalized moment tensor components range from -1 to 1
    lb = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, magnitude_min]
    ub = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, magnitude_max]

    def objective_function(params):
        try:
            # Create moment tensor from parameters
            mt = create_mt_from_normalized_components(params)
            
            # Calculate misfit for body waves and surface waves
            misfit_value_bw = misfit_bw(data_bw, greens_bw.select(origin), mt, optimization_level=0)
            misfit_value_sw = misfit_sw(data_sw, greens_sw.select(origin), mt, optimization_level=0)
            
            # Combined misfit - ensure scalar conversion
            misfit_sum = misfit_value_bw + misfit_value_sw
            
            # Convert to scalar using modern NumPy approach
            if hasattr(misfit_sum, 'item'):
                total_misfit = misfit_sum.item()
            else:
                total_misfit = float(misfit_sum)
                
            # Ensure we return a finite positive value
            if not np.isfinite(total_misfit) or total_misfit < 0:
                return 1e10
                
            return total_misfit
            
        except Exception as e:
            if rank == 0:
                print(f"Error in objective_function: {e}")
            return 1e10

    if rank == 0:
        print("Starting MPI-PSO full moment tensor optimization...")
    
    # Adjust swarm size for better MPI parallelization
    swarmsize = 100
    if swarmsize % size != 0:
        swarmsize = size * (swarmsize // size + 1)
    
    # Run PSO
    best_position, best_misfit, pso_time = pso_mpi_full_mt(
        objective_function, 
        lb, 
        ub,
        swarmsize=swarmsize, 
        maxiter=50, 
        stagnation_limit=10,
        stagnation_threshold=1e-4
    )
    
    # Post-process results (only on root process)
    if rank == 0:
        # Create best-fitting moment tensor
        best_mt = create_mt_from_normalized_components(best_position)
        
        # Extract parameters
        best_Mw = float(best_position[6])
        best_M0 = to_M0(best_Mw)
        
        # Calculate final misfit
        misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, optimization_level=0)
        misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, optimization_level=0)
        
        # Use modern NumPy scalar extraction
        if hasattr(misfit_bw_value, 'item'):
            misfit_bw_scalar = misfit_bw_value.item()
        else:
            misfit_bw_scalar = float(misfit_bw_value)
            
        if hasattr(misfit_sw_value, 'item'):
            misfit_sw_scalar = misfit_sw_value.item()
        else:
            misfit_sw_scalar = float(misfit_sw_value)
            
        final_misfit = misfit_bw_scalar + misfit_sw_scalar
        
        # Get lune parameters with corrected strike/dip/rake calculation
        lune_dict = mt_to_lune_parameters(best_mt)
        
        # Print diagnostic information
        print(f"Best solution parameters:")
        print(f"  Moment tensor components: {best_position[:6]}")
        print(f"  Magnitude: Mw {best_Mw:.2f}")
        print(f"  Strike: {lune_dict['kappa']:.1f}°")
        print(f"  Dip: {lune_dict['sigma']:.1f}°") 
        print(f"  Rake: {lune_dict['h']:.1f}°")
        print(f"  Fault Type: {lune_dict['fault_type']}")
        print(f"  Eigenvalues: {lune_dict['eigenvals']}")
        
        # Get moment tensor dictionary
        try:
            mt_dict = best_mt.as_dict() if hasattr(best_mt, 'as_dict') else {}
        except:
            mt_dict = {}
        
        # Create solution dictionary
        best_solution = merge_dicts(
            mt_dict,
            lune_dict,
            {'M0': float(best_M0)},
            {'Mw': float(best_Mw)},
            {'misfit_value': float(final_misfit)},
            {'computation_time': float(pso_time)},
            {'num_processes': size},
            {'optimization_method': 'PSO_full_moment_tensor'},
            origin.as_dict() if hasattr(origin, 'as_dict') else {}
        )
        
        return best_solution, best_mt, final_misfit, pso_time
    else:
        return None, None, None, pso_time

def main():
    """
    Main function to execute MPI-based full moment tensor inversion using PSO.
    """
    try:
        if rank == 0:
            overall_start_time = time.time()
            print("Starting Full Moment Tensor PSO Inversion...")
        
        # Define paths and parameters
        path_data = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/*.[zrt]')
        path_weights = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/weights.dat')
        event_id = '20090407201255351'
        model = 'ak135'
        
        # Magnitude search range
        magnitude_min = 4.0
        magnitude_max = 5.5
        initial_magnitude = (magnitude_min + magnitude_max) / 2

        # Data processing for body waves
        process_bw = ProcessData(
            filter_type='Bandpass',
            freq_min=0.1,
            freq_max=0.333,
            pick_type='taup',
            taup_model=model,
            window_type='body_wave',
            window_length=15.0,
            capuaf_file=path_weights,
        )

        # Data processing for surface waves
        process_sw = ProcessData(
            filter_type='Bandpass',
            freq_min=0.025,
            freq_max=0.0625,
            pick_type='taup',
            taup_model=model,
            window_type='surface_wave',
            window_length=150.0,
            capuaf_file=path_weights,
        )

        # Misfit functions
        misfit_bw = Misfit(
            norm='L2',
            time_shift_min=-2.0,
            time_shift_max=+2.0,
            time_shift_groups=['ZR'],
        )

        misfit_sw = Misfit(
            norm='L2',
            time_shift_min=-10.0,
            time_shift_max=+10.0,
            time_shift_groups=['ZR', 'T'],
        )

        # Parse station codes
        station_id_list = parse_station_codes(path_weights)

        # Define origin
        origin = Origin({
            'time': '2009-04-07T20:12:55.000000Z',
            'latitude': 61.454200744628906,
            'longitude': -149.7427978515625,
            'depth_in_m': 33033.599853515625,
        })
        
        # Source time function
        wavelet = Trapezoid(magnitude=initial_magnitude)

        # Data I/O
        if rank == 0:
            print('Reading data...')
        data = read(path_data, format='sac', event_id=event_id, 
                   station_id_list=station_id_list, tags=['units:m', 'type:velocity'])
        data.sort_by_distance()
        stations = data.get_stations()

        if rank == 0:
            print('Processing data...')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        if rank == 0:
            print('Reading Greens functions...')
        greens = download_greens_tensors(stations, origin, model)

        if rank == 0:
            print('Processing Greens functions...')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

        if rank == 0:
            print('Running PSO-based full moment tensor inversion...')

        # Run PSO-based inversion
        best_solution, best_mt, final_misfit, pso_time = pso_full_moment_tensor_inversion(
            data_bw, data_sw, greens_bw, greens_sw, 
            misfit_bw, misfit_sw, origin, stations, 
            event_id, process_bw, process_sw,
            magnitude_min=magnitude_min, 
            magnitude_max=magnitude_max
        )
        
        # Post-processing (root process only)
        if rank == 0:
            magnitude_val = float(best_solution['Mw'])
            M0_val = float(best_solution['M0'])
            
            print(f'Final misfit value: {float(final_misfit):.6f}')
            print(f'Best magnitude: Mw {magnitude_val:.2f}')
            print(f'Scalar moment: M0 {M0_val:.2e} N·m')
            print(f'PSO computation time: {pso_time:.2f} seconds')
            
            # Save results
            print('Saving results...')
            save_json(event_id + '_PSO_FullMT_solution.json', best_solution)
            
            # Generate plots
            filename_base = f"{event_id}_PSO_FullMT_Mw{magnitude_val:.2f}"
            
            print('Generating figures...')
            try:
                plot_data_greens2(filename_base + '_waveforms.png', 
                                data_bw, data_sw, greens_bw, greens_sw, 
                                process_bw, process_sw, misfit_bw, misfit_sw, 
                                stations, origin, best_mt, best_solution)
                
                plot_beachball(filename_base + '_beachball.png', 
                             best_mt, stations, origin)
                
                print('Full moment tensor PSO inversion completed successfully!')
                
            except Exception as e:
                print(f"Error during plotting: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        if rank == 0:
            print(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
