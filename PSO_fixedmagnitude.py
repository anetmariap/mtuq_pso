#!/usr/bin/env python

import os
import numpy as np
import pickle
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_dc
from mtuq.grid import Grid, UnstructuredGrid
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

def to_Mw(M0):
    """Convert seismic moment (M0) to moment magnitude (Mw)."""
    return (2 / 3) * (np.log10(M0) - 9.1)

def to_M0(Mw):
    """Convert moment magnitude (Mw) to seismic moment (M0)."""
    return 10 ** (1.5 * Mw + 9.1)

def normalize_sdr(strike, dip, rake):
    """
    Normalize strike, dip, and rake to standard ranges:
    - strike: 0-360 degrees
    - dip: 0-90 degrees
    - rake: -90 to 90 degrees
    
    If dip > 90, finds the equivalent representation with dip <= 90
    """
    strike = float(strike)
    dip = float(dip)
    rake = float(rake)
    
    # Normalize strike to 0-360
    strike = strike % 360
    
    # If dip > 90, find the conjugate fault plane
    if dip > 90:
        # Calculate conjugate fault plane
        strike = (strike + 180) % 360
        dip = 180 - dip
        rake = -rake  # This flips the rake sign
    
    # Ensure rake is between -90 and 90
    if rake < -90:
        rake += 180
        strike = (strike + 180) % 360
        rake = -rake
    elif rake > 90:
        rake -= 180
        strike = (strike + 180) % 360
        rake = -rake
        
    return strike, dip, rake

def create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=None, magnitude=None):
    """
    Create a MomentTensor object from strike, dip, and rake angles.
    Always normalize the angles first to ensure valid ranges.
    """
    strike, dip, rake = normalize_sdr(strike, dip, rake)
    
    # Convert angles from degrees to radians
    strike_rad = np.deg2rad(strike)
    dip_rad = np.deg2rad(dip)
    rake_rad = np.deg2rad(rake)

    # Calculate the moment tensor components
    sin_dip = np.sin(dip_rad)
    cos_dip = np.cos(dip_rad)
    sin_rake = np.sin(rake_rad)
    cos_rake = np.cos(rake_rad)
    sin_strike = np.sin(strike_rad)
    cos_strike = np.cos(strike_rad)
    sin_2dip = np.sin(2 * dip_rad)

    # Compute normalized tensor components
    Mrr = sin_2dip * sin_rake
    Mtt = -sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * sin_strike**2
    Mpp = sin_dip * cos_rake * np.sin(2 * strike_rad) - sin_2dip * sin_rake * cos_strike**2
    Mrt = -cos_dip * cos_rake * cos_strike - np.cos(2 * dip_rad) * sin_rake * sin_strike
    Mrp = -cos_dip * cos_rake * sin_strike + np.cos(2 * dip_rad) * sin_rake * cos_strike
    Mtp = sin_dip * cos_rake * np.cos(2 * strike_rad) + 0.5 * sin_2dip * sin_rake * np.sin(2 * strike_rad)

    # Create tensor in MTUQ order
    tensor = np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

    # Calculate scalar moment for proper scaling
    if magnitude is not None:
        scalar_moment = to_M0(magnitude)

    if scalar_moment is not None:
        # Scale the tensor properly by scalar moment
        mt = MomentTensor(tensor * scalar_moment)
    else:
        mt = MomentTensor(tensor)

    return mt, strike, dip, rake  # Return normalized angles along with moment tensor

def update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub):
    """
    Update swarm positions and velocities for PSO.
    """
    r1 = np.random.uniform(0, 1, swarm.shape)
    r2 = np.random.uniform(0, 1, swarm.shape)
    velocity = 0.5 * velocity + 1.5 * r1 * (personal_best_position - swarm) + 1.5 * r2 * (global_best_position - swarm)
    swarm = np.clip(swarm + velocity, lb, ub)
    return swarm, velocity

def pso_custom(objective_function, lb, ub, data_bw=None, data_sw=None, greens_bw=None, greens_sw=None, 
               process_bw=None, process_sw=None, misfit_bw=None, misfit_sw=None, origin=None, stations=None,
               event_id="event", save_iterations=False, swarmsize=100, maxiter=50, 
               stagnation_limit=10, stagnation_threshold=1e-4, debug=False):
    """
    Custom Particle Swarm Optimization (PSO) implementation with stagnation detection
    and option to save intermediate waveform plots.
    """
    dim = len(lb)
    swarm = np.random.uniform(low=lb, high=ub, size=(swarmsize, dim))
    velocity = np.random.uniform(low=-np.abs(np.array(ub) - np.array(lb)), high=np.abs(np.array(ub) - np.array(lb)), size=(swarmsize, dim))
    personal_best_position = np.copy(swarm)
    personal_best_value = np.array([objective_function(p) for p in personal_best_position])
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]
    global_best_value = personal_best_value[global_best_index]
    
    # For stagnation detection
    best_history = [global_best_value]
    stagnation_counter = 0
    
    # Create output directory for iteration plots if needed
    if save_iterations:
        iter_dir = f"{event_id}_iterations"
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)

    for i in range(maxiter):
        if debug:
            print(f"Iteration {i + 1}/{maxiter}, Best Value: {float(global_best_value)}")

        swarm, velocity = update_swarm(swarm, velocity, personal_best_position, global_best_position, lb, ub)
        current_values = np.array([objective_function(p) for p in swarm])

        # Update personal bests
        update_indices = np.where(current_values < personal_best_value)[0]
        for idx in update_indices:
            personal_best_position[idx] = swarm[idx]
            personal_best_value[idx] = current_values[idx]

        # Update global best
        prev_global_best = global_best_value
        if np.min(personal_best_value) < global_best_value:
            global_best_index = np.argmin(personal_best_value)
            global_best_position = personal_best_position[global_best_index]
            global_best_value = personal_best_value[global_best_index]
            
            # Save waveform plot for this iteration if improved and save_iterations is True
            if save_iterations and data_bw is not None:
                try:
                    # Get the raw position values
                    raw_strike = float(global_best_position[0])
                    raw_dip = float(global_best_position[1])
                    raw_rake = float(global_best_position[2])
                    
                    # Normalize angles to standard ranges
                    strike, dip, rake = normalize_sdr(raw_strike, raw_dip, raw_rake)
                    
                    scalar_moment = to_M0(4.5)  # Assuming fixed magnitude of 4.5
                    best_mt, _, _, _ = create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=scalar_moment)
                    
                    # Calculate misfit in the SAME way as the plotting function does
                    misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, optimization_level=0)
                    misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, optimization_level=0)
                    total_plot_misfit = float(misfit_bw_value + misfit_sw_value)  # Convert to Python float
                    
                    # Save iteration solution
                    lune_dict = {
                        'rho': float(scalar_moment),
                        'v': 0.0,
                        'w': 0.0,
                        'kappa': float(strike),  # Use normalized strike
                        'sigma': float(dip),     # Use normalized dip
                        'h': float(np.sin(np.deg2rad(rake))),  # Use normalized rake
                    }
                    
                    mt_dict = best_mt.as_dict()
                    iter_solution = merge_dicts(
                        mt_dict,
                        lune_dict,
                        {'M0': float(scalar_moment)},
                        {'Mw': 4.5},  # Fixed magnitude
                        {'fixed_Mw': 4.5},
                        {'misfit_value': float(total_plot_misfit)},  # Use the correctly calculated misfit
                        {'strike': float(strike)},  # Add normalized strike
                        {'dip': float(dip)},        # Add normalized dip
                        {'rake': float(rake)},      # Add normalized rake
                        origin.as_dict() if hasattr(origin, 'as_dict') else origin
                    )
                    
                    # Save the solution to track progress
                    save_json(f"{iter_dir}/{event_id}_iteration_{i+1}_solution.json", iter_solution)
                    
                    # Plot and save waveform comparison
                    # Format string with Python floats, not numpy arrays
                    figure_title = f"{event_id}_iteration_{i+1}_waveforms_strike{float(strike):.1f}_dip{float(dip):.1f}_rake{float(rake):.1f}"
                    plot_data_greens2(
                        f"{iter_dir}/{figure_title}.png", 
                        data_bw, data_sw, greens_bw, greens_sw, 
                        process_bw, process_sw, misfit_bw, misfit_sw, 
                        stations, origin, best_mt, iter_solution
                    )
                    
                    # Generate and save beachball plot with SDR information in filename
                    # Format string with Python floats, not numpy arrays
                    beachball_title = f"{event_id}_iteration_{i+1}_beachball_strike{float(strike):.1f}_dip{float(dip):.1f}_rake{float(rake):.1f}"
                    plot_beachball(
                        f"{iter_dir}/{beachball_title}.png", 
                        best_mt, stations, origin
                    )
                    
                    if debug:
                        print(f"  Iteration {i+1}, Misfit: {float(total_plot_misfit):.4f}, Strike: {float(strike):.1f}°, Dip: {float(dip):.1f}°, Rake: {float(rake):.1f}°")
                except Exception as e:
                    print(f"  Error saving iteration plot {i+1}: {e}")
                    import traceback
                    traceback.print_exc()  # Add this to get more detailed error info
        
        # Check for stagnation
        best_history.append(global_best_value)
        improvement = float(prev_global_best - global_best_value)
        
        if improvement < stagnation_threshold:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            
        if stagnation_counter >= stagnation_limit:
            if debug:
                print(f"Stopping early due to stagnation at iteration {i+1}")
            break

    # Return normalized best position
    # Convert original position to proper strike, dip, rake
    raw_strike = float(global_best_position[0])
    raw_dip = float(global_best_position[1])
    raw_rake = float(global_best_position[2])
    
    # Always normalize before returning
    norm_strike, norm_dip, norm_rake = normalize_sdr(raw_strike, raw_dip, raw_rake)
    normalized_position = np.array([norm_strike, norm_dip, norm_rake])
    
    return normalized_position, global_best_value

def pso_based_inversion(data_bw, data_sw, greens_bw, greens_sw, misfit_bw, misfit_sw, origin, stations, 
                        event_id, process_bw, process_sw, magnitude=4.5, save_iterations=False):
    """
    Perform moment tensor inversion using PSO with fixed magnitude.
    """
    # Define bounds for PSO (strike, dip, rake)
    lb = [0, 0, -90]  # Modified lower bound for rake: -90 instead of -180
    ub = [360, 90, 90]  # Modified upper bound for rake: 90 instead of 180

    # Calculate the reference moment explicitly for the given magnitude
    scalar_moment = to_M0(magnitude)

    def objective_function(params):
        try:
            # Convert numpy array elements to Python float
            strike = float(params[0])
            dip = float(params[1])
            rake = float(params[2])
            
            # Normalize angles to standard ranges
            strike, dip, rake = normalize_sdr(strike, dip, rake)
            
            mt, _, _, _ = create_mt_from_strike_dip_rake(strike, dip, rake, scalar_moment=scalar_moment)
            
        
            misfit_value_bw = misfit_bw(data_bw, greens_bw.select(origin), mt, optimization_level=0)
            misfit_value_sw = misfit_sw(data_sw, greens_sw.select(origin), mt, optimization_level=0)
            
            total_misfit = float(misfit_value_bw + misfit_value_sw)
            return total_misfit
        except Exception as e:
            print(f"Error in objective_function: {e}")
            import traceback
            traceback.print_exc()
            return 1e10

    # Run PSO with stagnation detection and save iteration plots
    best_position, best_misfit = pso_custom(
        objective_function, 
        lb, 
        ub,
        data_bw=data_bw,
        data_sw=data_sw,
        greens_bw=greens_bw,
        greens_sw=greens_sw,
        process_bw=process_bw,
        process_sw=process_sw,
        misfit_bw=misfit_bw,
        misfit_sw=misfit_sw,
        origin=origin,
        stations=stations,
        event_id=event_id,
        save_iterations=save_iterations,
        swarmsize=100, 
        maxiter=50, 
        stagnation_limit=10,
        stagnation_threshold=1e-4,
        debug=True
    )

    # Extract best-fitting parameters - already normalized by pso_custom
    best_strike = float(best_position[0])
    best_dip = float(best_position[1])
    best_rake = float(best_position[2])
    
    best_M0 = float(scalar_moment)
    best_Mw = float(magnitude)  # Fixed at given magnitude
    
    # Create moment tensor with normalized angles
    best_mt, best_strike, best_dip, best_rake = create_mt_from_strike_dip_rake(
        best_strike, best_dip, best_rake, scalar_moment=best_M0
    )

    # Calculate final misfit using the exact same method as in plot_data_greens2
    misfit_bw_value = misfit_bw(data_bw, greens_bw.select(origin), best_mt, optimization_level=0)
    misfit_sw_value = misfit_sw(data_sw, greens_sw.select(origin), best_mt, optimization_level=0)
    final_plot_misfit = float(misfit_bw_value + misfit_sw_value)

    # Create output dictionary with normalized angles
    lune_dict = {
        'rho': float(best_M0),
        'v': 0.0,
        'w': 0.0,
        'kappa': float(best_strike),  # Normalized strike
        'sigma': float(best_dip),     # Normalized dip
        'h': float(np.sin(np.deg2rad(best_rake))),  # Based on normalized rake
    }

    mt_dict = best_mt.as_dict()
    best_solution = merge_dicts(
        mt_dict,
        lune_dict,
        {'M0': float(best_M0)},
        {'Mw': float(best_Mw)},
        {'fixed_Mw': float(magnitude)},
        {'misfit_value': float(final_plot_misfit)},  # Use the correctly calculated plot misfit
        {'strike': float(best_strike)},  # Add normalized strike
        {'dip': float(best_dip)},        # Add normalized dip
        {'rake': float(best_rake)},      # Add normalized rake
        origin.as_dict() if hasattr(origin, 'as_dict') else origin
    )

    return best_solution, best_mt, final_plot_misfit

if __name__ == '__main__':
    try:
        # Define paths and parameters
        path_data = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/*.[zrt]')
        path_weights = fullpath('/home/anetmariap/mtuq/data/examples/20090407201255351/weights.dat')
        event_id = '20090407201255351'
        model = 'ak135'
        magnitude = 4.5  # Fixed magnitude
        save_iterations = True  # Set to True to save iteration plots

        # Define data processing for body waves
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

        # Define data processing for surface waves
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

        # Define misfit functions
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

        # Define source time function
        wavelet = Trapezoid(magnitude=magnitude)

        print('Reading data...\n')
        data = read(path_data, format='sac', event_id=event_id, station_id_list=station_id_list, tags=['units:m', 'type:velocity'])
        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)

        print('Running PSO-based inversion with fixed magnitude...\n')

        # Run PSO-based inversion
        best_solution, best_mt, final_misfit = pso_based_inversion(
            data_bw, data_sw, greens_bw, greens_sw, 
            misfit_bw, misfit_sw, origin, stations, 
            event_id, process_bw, process_sw,
            magnitude=magnitude, 
            save_iterations=save_iterations
        )
        
        # Double check that the result is properly normalized
        strike_val = float(best_solution['strike'])
        dip_val = float(best_solution['dip'])
        rake_val = float(best_solution['rake'])
        
        print(f'\nFinal misfit value: {float(final_misfit)}')
        print(f'Best strike: {strike_val:.1f}°, dip: {dip_val:.1f}°, rake: {rake_val:.1f}°')
        
        if rake_val < -90 or rake_val > 90 or dip_val > 90:
            print(f"WARNING: Final values (strike={strike_val:.1f}°, dip={dip_val:.1f}°, rake={rake_val:.1f}°) may be outside valid ranges")
            print("Re-normalizing final solution...")
            norm_strike, norm_dip, norm_rake = normalize_sdr(strike_val, dip_val, rake_val)
            print(f"Corrected solution: Strike={norm_strike:.1f}°, Dip={norm_dip:.1f}°, Rake={norm_rake:.1f}°")
            
            # Update the solution before saving and plotting
            best_solution['strike'] = float(norm_strike)
            best_solution['dip'] = float(norm_dip) 
            best_solution['rake'] = float(norm_rake)
            best_solution['kappa'] = float(norm_strike)
            best_solution['sigma'] = float(norm_dip)
            best_solution['h'] = float(np.sin(np.deg2rad(norm_rake)))
            
            # Create new moment tensor with normalized values
            best_mt, _, _, _ = create_mt_from_strike_dip_rake(norm_strike, norm_dip, norm_rake, scalar_moment=best_solution['M0'])
            
            strike_val = norm_strike
            dip_val = norm_dip
            rake_val = norm_rake

        print('Saving results...\n')
        save_json(event_id + '_PSO_solution.json', best_solution)

        # Include SDR values in final figure filenames
        final_filename_base = f"{event_id}_PSO_strike{strike_val:.1f}_dip{dip_val:.1f}_rake{rake_val:.1f}"
        
        print('Generating final figures...\n')
        try:
            plot_data_greens2(final_filename_base + '_waveforms.png', data_bw, data_sw, greens_bw, greens_sw, process_bw, process_sw, misfit_bw, misfit_sw, stations, origin, best_mt, best_solution)
            plot_beachball(final_filename_base + '_beachball.png', best_mt, stations, origin)
            print('\nFinished\n')
        except Exception as e:
            print(f"Error during plotting: {e}")
            import traceback
            traceback.print_exc()  # Add this to get more detailed error info

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
