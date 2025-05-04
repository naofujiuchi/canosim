#%%
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

# create a folder for the simulation results
os.makedirs('simulation', exist_ok=True)

# Parameters in real 
small_leaf_length = 8  # Semi-major axis length (small leaf length) [cm]
small_leaf_width = 4.5  # Semi-minor axis length (small leaf width) [cm]
a = small_leaf_length / 2  # Semi-major axis length (small leaf length) [cm]
b = small_leaf_width / 2  # Semi-minor axis length (small leaf width) [cm]
small_leaf_area = a * b * np.pi # leaf area of the small leaf [cm2]
fruit_diameter = 1.9 # diameter of the ball (fruit diameter) [cm]
radius = fruit_diameter / 2 # radius of the ball (fruit diameter / 2) [cm]
analysis_width = 50 # range of the analysis [cm]
analysis_height = 206 # range of the analysis [cm]
leaf_area_min = 0 # minimum leaf area in 50cm width and 1.5m height [m2]
leaf_area_max = 1.5 # maximum leaf area in 50cm width and 1.5m height [m2]
fruit_number_min = 0 # minimum number of the fruit in 50cm width and 1.5m height
fruit_number_max = 250 # maximum number of the fruit in 50cm width and 1.5m height
interval = 20 # plant interval [cm]. Rule: leaves and fruits can be in the circle at a radius of 20 cm from the plant stem
pixel_per_cm = 33 # pixel per cm
leaf_tilt_angle_mean = -54 # leaf tilt angle [degree]
leaf_tilt_angle_std = 18 # leaf tilt angle [degree]

# Parameters in simulation
analysis_width_simulation = 60 # range of the analysis [cm]
analysis_height_simulation = 206 # range of the analysis [cm]
radius_plant_circle = 20 # radius of the plant circle [cm]
leaf_area_min_simulation = 0 * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # minimum leaf area in 50cm width and 1.5m height [m2]
leaf_area_max_simulation = leaf_area_max * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # maximum leaf area in 50cm width and 1.5m height [m2]
fruit_number_min_simulation = 0 * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # minimum number of the fruit in 50cm width and 1.5m height
fruit_number_max_simulation = fruit_number_max * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # maximum number of the fruit in 50cm width and 1.5m height

# small_leaf_area = 21.4 # leaf area of the small leaf [cm2]

# grid of leaf_area and fruit_number
leaf_area_grid = np.linspace(leaf_area_min, leaf_area_max, 11)
fruit_number_grid = np.linspace(fruit_number_min, fruit_number_max, 11)
leaf_area_simulation_grid = np.linspace(leaf_area_min_simulation, leaf_area_max_simulation, 11)
fruit_number_simulation_grid = np.linspace(fruit_number_min_simulation, fruit_number_max_simulation, 11)

df_result = pd.DataFrame(columns=['i', 'j', 'analysis_width', 'analysis_height', 'leaf_area', 'fruit_number', 'analysis_width_simulation', 'analysis_height_simulation', 'leaf_area_simulation', 'fruit_number_simulation', 'small_leaf_number_simulation', 'fruit_number_detected_simulation'])

for i in [5]:
    for j in [5]:

        print(f'i: {i}, j: {j}')

        leaf_area = leaf_area_grid[i] # [m2]
        fruit_number = fruit_number_grid[j]
        # samll_leaf_number = leaf_area * 10000 / small_leaf_area # number of the small leaf

        leaf_area_simulation = leaf_area_simulation_grid[i]
        fruit_number_simulation = fruit_number_simulation_grid[j]
        small_leaf_number_simulation = leaf_area_simulation * 10000 / small_leaf_area # number of the small leaf

        # Create a grid of points
        theta = np.linspace(0, 2 * np.pi, 20)
        r = np.linspace(0, 1, 10)
        Theta, R = np.meshgrid(theta, r)

        # Parametric equations for the leaf shape
        X = a * R * np.cos(Theta)
        Y = b * R * np.sin(Theta)
        Z = np.full_like(X, 0)

        # Function to apply specific distribution for rotation and translation
        def transform_leaf(X, Y, Z, translation, angle_mean=leaf_tilt_angle_mean/90*(np.pi/2), angle_std=leaf_tilt_angle_std/90*(np.pi/2)):
            rotation_angle_x = 0
            rotation_angle_y = np.random.normal(angle_mean, angle_std)
            rotation_angle_z = np.random.uniform(0, 2*np.pi)

            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
                [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]
            ])
            
            Ry = np.array([
                [np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
                [0, 1, 0],
                [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]
            ])
            
            Rz = np.array([
                [np.cos(rotation_angle_z), -np.sin(rotation_angle_z), 0],
                [np.sin(rotation_angle_z), np.cos(rotation_angle_z), 0],
                [0, 0, 1]
            ])

            # Combined rotation matrix
            R = Rz @ Ry @ Rx

            # Apply rotation
            X_rotated = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
            Y_rotated = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
            Z_rotated = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z

            # Random translation
            X_translated = X_rotated + translation[0]
            Y_translated = Y_rotated + translation[1]
            Z_translated = Z_rotated + translation[2]

            return X_translated, Y_translated, Z_translated

        # Function to create a sphere
        def create_sphere(center, radius, resolution=10):
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            return x, y, z

        layout = go.Layout(autosize=False, width=analysis_width_simulation*4, height=analysis_height_simulation*4, margin=dict(l=0, r=0, t=0, b=0))
        fig = go.Figure(layout=layout)

        for k in range(int(small_leaf_number_simulation)):  # Add leaves
        # for i in range(10):  # Add leaves
            # translation = np.random.uniform(-analysis_range, analysis_range, size=3)
            location_r = np.random.uniform(0, radius_plant_circle, size=1)
            location_theta = np.random.uniform(0, 6*np.pi, size=1)
            if location_theta < np.pi:
                location_x = location_r * np.sin(location_theta)
            elif location_theta >= np.pi and location_theta < 3*np.pi:
                location_x = interval + location_r * np.sin(location_theta)
            elif location_theta >= 3*np.pi and location_theta < 5*np.pi:
                location_x = 2*interval + location_r * np.sin(location_theta)
            elif location_theta >= 5*np.pi and location_theta < 7*np.pi:
                location_x = 3*interval + location_r * np.sin(location_theta)
            location_y = analysis_width_simulation/2 + location_r * np.cos(location_theta)
            location_z = np.random.uniform(0, analysis_height_simulation, size=1)
            translation = np.array([location_x, location_y, location_z])    
            X1, Y1, Z1 = transform_leaf(X, Y, Z, translation=translation)
            # fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, colorscale='Greens', opacity=1, showscale=False))
            fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, colorscale=[[0,'Green'], [1,'Green']], opacity=1, showscale=False))    

        # Add balls
        for l in range(int(fruit_number_simulation)):  # Add 5 balls
        # for _ in range(1):  # Add 5 balls
            # center = np.random.uniform(0, analysis_range, size=3)
            location_r = np.random.uniform(0, radius_plant_circle, size=1)
            location_theta = np.random.uniform(0, 6*np.pi, size=1)
            if location_theta < np.pi:
                location_x = location_r * np.sin(location_theta)
            elif location_theta >= np.pi and location_theta < 3*np.pi:
                location_x = interval + location_r * np.sin(location_theta)
            elif location_theta >= 3*np.pi and location_theta < 5*np.pi:
                location_x = 2*interval + location_r * np.sin(location_theta)
            elif location_theta >= 5*np.pi and location_theta < 7*np.pi:
                location_x = 3*interval + location_r * np.sin(location_theta)
            location_y = analysis_width_simulation/2 + location_r * np.cos(location_theta)
            location_z = np.random.uniform(0, analysis_height_simulation, size=1)
            center = np.array([location_x, location_y, location_z])    
            x, y, z = create_sphere(center, radius)
            # fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Reds', opacity=1, showscale=False))
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0,'Red'], [1,'Red']], opacity=1, showscale=False))


        # def add_column_surface(fig, start_theta, end_theta, x_offset=0):
        #     theta = np.linspace(start_theta, end_theta, 20)
        #     z = np.linspace(0, analysis_height_simulation, 20)
        #     Theta, Z = np.meshgrid(theta, z)
        #     X = x_offset + radius_plant_circle * np.sin(Theta)
        #     Y = analysis_width_simulation / 2 + radius_plant_circle * np.cos(Theta)
        #     fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, 'Yellow'], [1, 'Yellow']], opacity=0.5, showscale=False))

        # def add_circle_plate(fig, theta_start, theta_end, z_value, x_offset=0):
        #     theta = np.linspace(theta_start, theta_end, 20)
        #     r = np.linspace(0, radius_plant_circle, 20)
        #     Theta, R = np.meshgrid(theta, r)
        #     X = x_offset + R * np.sin(Theta)
        #     Y = analysis_width_simulation / 2 + R * np.cos(Theta)
        #     Z = np.full_like(X, z_value)
        #     fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, 'Yellow'], [1, 'Yellow']], opacity=0.5, showscale=False))

        # # Add column surfaces
        # for i in range(4):
        #     if i == 0:
        #         add_column_surface(fig, i * np.pi, (i + 1) * np.pi, x_offset=i * interval)
        #     else:
        #         add_column_surface(fig, i * np.pi, (i + 1) * np.pi, x_offset=i * interval)

        # # Add plates at x = 0 and x = analysis_width_simulation
        # for x_value in [0, analysis_width_simulation]:
        #     y = np.linspace(10, 50, 20)
        #     z = np.linspace(0, analysis_height_simulation, 20)
        #     Y, Z = np.meshgrid(y, z)
        #     X = np.full_like(Y, x_value)
        #     fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, 'Yellow'], [1, 'Yellow']], opacity=0.5, showscale=False))

        # # Add circle plates on the edge of the columns
        # for i in range(4):
        #     add_circle_plate(fig, i * np.pi, (i + 1) * np.pi, 0, x_offset=i * interval)
        #     add_circle_plate(fig, i * np.pi, (i + 1) * np.pi, analysis_height_simulation, x_offset=i * interval)


        draw_color = 'Gray'
        draw_opacity = 0.2

        # Add column surface of the plant circle
        theta = np.linspace(0, np.pi, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Theta, Z = np.meshgrid(theta, z)
        X = radius_plant_circle * np.sin(Theta)
        Y = analysis_width_simulation/2 + radius_plant_circle * np.cos(Theta)
        Z = Z
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(np.pi, 3 * np.pi, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Theta, Z = np.meshgrid(theta, z)
        X = interval + radius_plant_circle * np.sin(Theta)
        Y = analysis_width_simulation/2 + radius_plant_circle * np.cos(Theta)
        Z = Z
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(3 * np.pi, 5 * np.pi, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Theta, Z = np.meshgrid(theta, z)
        X = 2 * interval + radius_plant_circle * np.sin(Theta)
        Y = analysis_width_simulation/2 + radius_plant_circle * np.cos(Theta)
        Z = Z
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(5 * np.pi, 6 * np.pi, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Theta, Z = np.meshgrid(theta, z)
        X = 3 * interval + radius_plant_circle * np.sin(Theta)
        Y = analysis_width_simulation/2 + radius_plant_circle * np.cos(Theta)
        Z = Z
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        # Add a plate of x = 0 and 10 <= y <= 50, x = analysis_width_simulation and 10 <= y <= 50. 0 <= z <= analysis_height_simulation
        y = np.linspace(10, 50, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        y = np.linspace(10, 50, 20)
        z = np.linspace(0, analysis_height_simulation, 20)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, analysis_width_simulation)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        # Add circle plates on the edge of the columns        
        theta = np.linspace(0, 1 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))
        
        theta = np.linspace(0, 1 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, analysis_height_simulation)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(np.pi, 3 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))
        
        theta = np.linspace(np.pi, 3 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, analysis_height_simulation)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(3 * np.pi, 5 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = 2 * interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))
        
        theta = np.linspace(3 * np.pi, 5 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = 2 * interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, analysis_height_simulation)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        theta = np.linspace(5 * np.pi, 6 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = 3 * interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, 0)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))
        
        theta = np.linspace(5 * np.pi, 6 * np.pi, 20)
        r = np.linspace(0, radius_plant_circle, 20)
        Theta, R = np.meshgrid(theta, r)
        X = 3 * interval + R * np.sin(Theta)
        Y = analysis_width_simulation/2 + R * np.cos(Theta)
        Z = np.full_like(X, analysis_height_simulation)
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0,draw_color], [1,draw_color]], opacity=draw_opacity, showscale=False))

        # Update layout to remove axis titles and values
        fig.update_layout(scene=dict(
            xaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            yaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            zaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_height_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            aspectratio=dict(x=1, y=1, z=analysis_height_simulation/analysis_width_simulation),
        ))

        fig.layout.scene.camera.projection.type = "orthographic"
            
        fig.write_html(f"simulation/projection_view_i_{i}_j_{j}.html")

#%%
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

# create a folder for the simulation results
os.makedirs('simulation', exist_ok=True)

# Parameters in real 
small_leaf_length = 8  # Semi-major axis length (small leaf length) [cm]
small_leaf_width = 4.5  # Semi-minor axis length (small leaf width) [cm]
a = small_leaf_length / 2  # Semi-major axis length (small leaf length) [cm]
b = small_leaf_width / 2  # Semi-minor axis length (small leaf width) [cm]
small_leaf_area = a * b * np.pi # leaf area of the small leaf [cm2]
fruit_diameter = 1.9 # diameter of the ball (fruit diameter) [cm]
radius = fruit_diameter / 2 # radius of the ball (fruit diameter / 2) [cm]
analysis_width = 50 # range of the analysis [cm]
analysis_height = 206 # range of the analysis [cm]
leaf_area_min = 0 # minimum leaf area in 50cm width and 1.5m height [m2]
leaf_area_max = 1.5 # maximum leaf area in 50cm width and 1.5m height [m2]
fruit_number_min = 0 # minimum number of the fruit in 50cm width and 1.5m height
fruit_number_max = 250 # maximum number of the fruit in 50cm width and 1.5m height
interval = 20 # plant interval [cm]. Rule: leaves and fruits can be in the circle at a radius of 20 cm from the plant stem
pixel_per_cm = 33 # pixel per cm
leaf_tilt_angle_mean = -54 # leaf tilt angle [degree]
leaf_tilt_angle_std = 18 # leaf tilt angle [degree]

# Parameters in simulation
analysis_width_simulation = 60 # range of the analysis [cm]
analysis_height_simulation = 206 # range of the analysis [cm]
radius_plant_circle = 20 # radius of the plant circle [cm]
leaf_area_min_simulation = 0 * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # minimum leaf area in 50cm width and 1.5m height [m2]
leaf_area_max_simulation = leaf_area_max * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # maximum leaf area in 50cm width and 1.5m height [m2]
fruit_number_min_simulation = 0 * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # minimum number of the fruit in 50cm width and 1.5m height
fruit_number_max_simulation = fruit_number_max * (analysis_width_simulation * analysis_height_simulation) / (analysis_width * analysis_height) # maximum number of the fruit in 50cm width and 1.5m height

# small_leaf_area = 21.4 # leaf area of the small leaf [cm2]

# grid of leaf_area and fruit_number
leaf_area_grid = np.linspace(leaf_area_min, leaf_area_max, 11)
fruit_number_grid = np.linspace(fruit_number_min, fruit_number_max, 11)
leaf_area_simulation_grid = np.linspace(leaf_area_min_simulation, leaf_area_max_simulation, 11)
fruit_number_simulation_grid = np.linspace(fruit_number_min_simulation, fruit_number_max_simulation, 11)

df_result = pd.DataFrame(columns=['i', 'j', 'analysis_width', 'analysis_height', 'leaf_area', 'fruit_number', 'analysis_width_simulation', 'analysis_height_simulation', 'leaf_area_simulation', 'fruit_number_simulation', 'small_leaf_number_simulation', 'fruit_number_detected_simulation'])

for i in range(len(leaf_area_simulation_grid)):
    for j in range(len(fruit_number_simulation_grid)):

        print(f'i: {i}, j: {j}')

        leaf_area = leaf_area_grid[i] # [m2]
        fruit_number = fruit_number_grid[j]
        # samll_leaf_number = leaf_area * 10000 / small_leaf_area # number of the small leaf

        leaf_area_simulation = leaf_area_simulation_grid[i]
        fruit_number_simulation = fruit_number_simulation_grid[j]
        small_leaf_number_simulation = leaf_area_simulation * 10000 / small_leaf_area # number of the small leaf

        # Create a grid of points
        theta = np.linspace(0, 2 * np.pi, 20)
        r = np.linspace(0, 1, 10)
        Theta, R = np.meshgrid(theta, r)

        # Parametric equations for the leaf shape
        X = a * R * np.cos(Theta)
        Y = b * R * np.sin(Theta)
        Z = np.full_like(X, 0)

        # Function to apply specific distribution for rotation and translation
        def transform_leaf(X, Y, Z, translation, angle_mean=leaf_tilt_angle_mean/90*(np.pi/2), angle_std=leaf_tilt_angle_std/90*(np.pi/2)):
            rotation_angle_x = 0
            rotation_angle_y = np.random.normal(angle_mean, angle_std)
            rotation_angle_z = np.random.uniform(0, 2*np.pi)

            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_angle_x), -np.sin(rotation_angle_x)],
                [0, np.sin(rotation_angle_x), np.cos(rotation_angle_x)]
            ])
            
            Ry = np.array([
                [np.cos(rotation_angle_y), 0, np.sin(rotation_angle_y)],
                [0, 1, 0],
                [-np.sin(rotation_angle_y), 0, np.cos(rotation_angle_y)]
            ])
            
            Rz = np.array([
                [np.cos(rotation_angle_z), -np.sin(rotation_angle_z), 0],
                [np.sin(rotation_angle_z), np.cos(rotation_angle_z), 0],
                [0, 0, 1]
            ])

            # Combined rotation matrix
            R = Rz @ Ry @ Rx

            # Apply rotation
            X_rotated = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z
            Y_rotated = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z
            Z_rotated = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * Z

            # Random translation
            X_translated = X_rotated + translation[0]
            Y_translated = Y_rotated + translation[1]
            Z_translated = Z_rotated + translation[2]

            return X_translated, Y_translated, Z_translated

        # Function to create a sphere
        def create_sphere(center, radius, resolution=10):
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            return x, y, z

        # Create the plot
        layout = go.Layout(autosize=False, width=analysis_width_simulation, height=analysis_height_simulation, margin=dict(l=0, r=0, t=0, b=0))
        fig = go.Figure(layout=layout)

        for k in range(int(small_leaf_number_simulation)):  # Add leaves
        # for i in range(10):  # Add leaves
            # translation = np.random.uniform(-analysis_range, analysis_range, size=3)
            location_r = np.random.uniform(0, radius_plant_circle, size=1)
            location_theta = np.random.uniform(0, 6*np.pi, size=1)
            if location_theta < np.pi:
                location_x = location_r * np.sin(location_theta)
            elif location_theta >= np.pi and location_theta < 3*np.pi:
                location_x = interval + location_r * np.sin(location_theta)
            elif location_theta >= 3*np.pi and location_theta < 5*np.pi:
                location_x = 2*interval + location_r * np.sin(location_theta)
            elif location_theta >= 5*np.pi and location_theta < 7*np.pi:
                location_x = 3*interval + location_r * np.sin(location_theta)
            location_y = analysis_width_simulation/2 + location_r * np.cos(location_theta)
            location_z = np.random.uniform(0, analysis_height_simulation, size=1)
            translation = np.array([location_x, location_y, location_z])    
            X1, Y1, Z1 = transform_leaf(X, Y, Z, translation=translation)
            # fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, colorscale='Greens', opacity=1, showscale=False))
            fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, colorscale=[[0,'Green'], [1,'Green']], opacity=1, showscale=False))    

        # Add balls
        for l in range(int(fruit_number_simulation)):  # Add 5 balls
        # for _ in range(1):  # Add 5 balls
            # center = np.random.uniform(0, analysis_range, size=3)
            location_r = np.random.uniform(0, radius_plant_circle, size=1)
            location_theta = np.random.uniform(0, 6*np.pi, size=1)
            if location_theta < np.pi:
                location_x = location_r * np.sin(location_theta)
            elif location_theta >= np.pi and location_theta < 3*np.pi:
                location_x = interval + location_r * np.sin(location_theta)
            elif location_theta >= 3*np.pi and location_theta < 5*np.pi:
                location_x = 2*interval + location_r * np.sin(location_theta)
            elif location_theta >= 5*np.pi and location_theta < 7*np.pi:
                location_x = 3*interval + location_r * np.sin(location_theta)
            location_y = analysis_width_simulation/2 + location_r * np.cos(location_theta)
            location_z = np.random.uniform(0, analysis_height_simulation, size=1)
            center = np.array([location_x, location_y, location_z])    
            x, y, z = create_sphere(center, radius)
            # fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Reds', opacity=1, showscale=False))
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0,'Red'], [1,'Red']], opacity=1, showscale=False))

        # Update layout to remove axis titles and values
        fig.update_layout(scene=dict(
            xaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            yaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            zaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_height_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            aspectratio=dict(x=1, y=1, z=1),  # Set aspect ratio to 1:1:1 for a cube
            camera=dict(
                eye=dict(x=0, y=0, z=2)  # Set the camera to look from the side at z = 0
            )
        ))
            
        fig.layout.scene.camera.projection.type = "orthographic"
        
        # Export the plot as a static image
        output_image_width = pixel_per_cm * analysis_width_simulation
        output_image_height = output_image_width
        fig.write_image(f"simulation/upper_projection_view_i_{i}_j_{j}.png", width=output_image_width, height=output_image_height)

        # Update layout to remove axis titles and values
        fig.update_layout(scene=dict(
            xaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            yaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_width_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            zaxis=dict(
                title='',
                showticklabels=False,
                range=[0, analysis_height_simulation]
                # range=[-2*analysis_range, 2*analysis_range]
            ),
            aspectratio=dict(x=1, y=1, z=2),  # Set aspect ratio to 1:1:1 for a cube
            camera=dict(
                eye=dict(x=0, y=1, z=0)  # Set the camera to look from the side at x = 0
            )
        ))
            
        fig.layout.scene.camera.projection.type = "orthographic"

        # Export the plot as a static image
        output_image_width = pixel_per_cm * analysis_width_simulation
        output_image_height = pixel_per_cm * analysis_height_simulation
        fig.write_image(f"simulation/projection_view_i_{i}_j_{j}.png", width=output_image_width, height=output_image_height)

        # # Show the plot
        # fig.show()

        # Load the image
        image = cv2.imread(f'simulation/projection_view_i_{i}_j_{j}.png')

        # Check if the image was loaded successfully
        if image is None:
            print("Error: Image not loaded. Please check the file path.")
            exit()

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV
        # Note: Red color can wrap around the hue value, so we need two ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.add(mask1, mask2)

        # Apply a series of dilations and erosions to remove any small blobs left in the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fruit_count = 0

        # Loop over the contours
        for contour in contours:
            # Calculate the area and perimeter of the contour
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # Check if the contour is circular enough
            if circularity > 0.7 and area > 100:  # Adjust the threshold as needed
                # Approximate the contour to a circle
                ((x, y), radius_contour) = cv2.minEnclosingCircle(contour)
                
                # Only consider contours with a significant radius
                if radius_contour > 10:
                    fruit_count += 1
                    # Draw the circle on the original image
                    cv2.circle(image, (int(x), int(y)), int(radius_contour), (0, 255, 0), 2)
                    # Optionally, draw the center of the circle
                    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 3)

        print('Number of detected fruits:', fruit_count)
        fruit_count_real = fruit_count * (analysis_width * analysis_height) / (analysis_width_simulation * analysis_height_simulation)

        # Display the result
        # plt.imshow(image)
        cv2.imwrite(f'simulation/red_detection_i_{i}_j_{j}.png', image)

        image = cv2.imread(f'simulation/projection_view_i_{i}_j_{j}.png')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Define the range for green color in HSV
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        # Create masks for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Count the number of non-zero pixels in the mask
        leaf_pixels = cv2.countNonZero(mask)
        leaf_pixels_million = leaf_pixels / 1000000
        print(f'i: {i}, j: {j}, leaf_pixels_million: {leaf_pixels_million}')
        leaf_pixels_million_real = leaf_pixels_million * (analysis_width * analysis_height) / (analysis_width_simulation * analysis_height_simulation)

        df_result = pd.concat([df_result, pd.DataFrame({'i': i, 
                                                        'j': j, 
                                                        'analysis_width': analysis_width, 
                                                        'analysis_height': analysis_height, 
                                                        'leaf_area': leaf_area, 
                                                        'fruit_number': fruit_number, 
                                                        'analysis_width_simulation': analysis_width_simulation, 
                                                        'analysis_height_simulation': analysis_height_simulation, 
                                                        'leaf_area_simulation': leaf_area_simulation, 
                                                        'fruit_number_simulation': fruit_number_simulation, 
                                                        'small_leaf_number_simulation': small_leaf_number_simulation, 
                                                        'fruit_number_detected_simulation': fruit_count,
                                                        'leaf_pixels_million_simulation': leaf_pixels_million,
                                                        'fruit_count_real': fruit_count_real,
                                                        'leaf_pixels_million_real': leaf_pixels_million_real}, 
                                                        index=[0])], ignore_index=True)
df_result.to_csv('simulation_result.csv', index=False)


#%%
import cv2
import numpy as np

# Load the image
image = cv2.imread(f'archive_20250406/simulation/projection_view_i_5_j_5.png')
# Check if the image was loaded successfully
if image is None:
    print("Error: Image not loaded. Please check the file path.")
    exit()

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for red color in HSV
# Note: Red color can wrap around the hue value, so we need two ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create masks for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.add(mask1, mask2)

# Apply a series of dilations and erosions to remove any small blobs left in the mask
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

fruit_count = 0

black_image = np.zeros_like(image)

# Loop over the contours
for contour in contours:
    # Calculate the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # Check if the contour is circular enough
    if circularity > 0.7 and area > 100:  # Adjust the threshold as needed
        # Approximate the contour to a circle
        ((x, y), radius_contour) = cv2.minEnclosingCircle(contour)
        
        # Only consider contours with a significant radius
        if radius_contour > 10:
            fruit_count += 1
            # Draw the circle on the original image
            cv2.circle(black_image, (int(x), int(y)), int(radius_contour), (255, 255, 255), 6)
            # Optionally, draw the center of the circle
            cv2.circle(black_image, (int(x), int(y)), 2, (255, 255, 255), 24)

cv2.imwrite('simulation/circle_detection_i_5_j_5.png', black_image)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

df_result = pd.read_csv('archive_20250406/simulation_result.csv')
df_result['fruit_occuluded_ratio'] = (df_result['fruit_number_simulation'] - df_result['fruit_number_detected_simulation']) / df_result['fruit_number_simulation']

# Make plot of leaf_area vs leaf_pixels_million_real
plt.figure(figsize=(3, 3), dpi=200)
plt.scatter(df_result['leaf_area'], df_result['leaf_pixels_million_real'], c=df_result['fruit_number'], marker='o')
plt.xlabel(r'Leaf area [m$^2$]')
plt.ylabel(r'Leaf pixels [10$^6$pixels]')
plt.colorbar(label='Fruit number')
# plt.legend()
# Set axes to start from zero
plt.xlim(left=0)
plt.ylim(bottom=0)
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
plt.show()

# Make plot of fruit_number vs fruit_count_real
plt.figure(figsize=(3, 3), dpi=200)
plt.scatter(df_result['fruit_number'], df_result['fruit_count_real'], c=df_result['leaf_area'], marker='o')
plt.xlabel(r'Fruit number')
plt.ylabel(r'Fruit detected number')
# plt.legend()
# Set axes to start from zero
plt.xlim(left=0, right=270)
plt.ylim(bottom=0, top=270)
plt.plot(np.linspace(0, 270, 100), np.linspace(0, 270, 100), color='black', linestyle='--')
# x axis tick: 50, 100, 150, 200, 250
plt.xticks([0, 50, 100, 150, 200, 250])
# y axis tick: 50, 100, 150, 200, 250
plt.yticks([0, 50, 100, 150, 200, 250])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
plt.colorbar(label='Leaf area [m$^2$]')
plt.show()


# Make 3d plot of leaf_pixels_million and fruit_count_real vs fruit_number
fig = px.scatter_3d(df_result, x='leaf_pixels_million_real', y='fruit_count_real', z='fruit_number')
# orthographic projection
fig.update_layout(scene=dict(
    xaxis_title='葉量[10^6 pixel]',
    yaxis_title='横から見える果実数',
    zaxis_title='樹上果実数',    
    camera=dict(
        projection=dict(type='orthographic')
    )
))

fig.show()

# # export as html
# fig.write_html('simulation_result_leafpixels.html')

# plot of fruit_occuluded_ratio vs leaf_pixels_million
plt.figure(figsize=(3, 3), dpi=200)
# Create the scatter plot
scatter = plt.scatter(
    df_result['leaf_pixels_million_real'], 
    df_result['fruit_occuluded_ratio'], 
    c=df_result['fruit_number'], 
    cmap='viridis',  # You can choose any colormap you like
    marker='o'
)
# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Fruit Number')  # Label for the colorbar
# Add labels and adjust layout
plt.xlabel(r'Leaf pixels [10$^6$pixels]')
plt.ylabel(r'Fruit occlusion ratio')
plt.gca().set_box_aspect(1)
plt.show()

# plot of fruit_occuluded_ratio vs leaf_area
plt.figure(figsize=(3, 3), dpi=200)
# Create the scatter plot
scatter = plt.scatter(
    df_result['leaf_area'], 
    df_result['fruit_occuluded_ratio'], 
    c=df_result['fruit_number'], 
    cmap='viridis',  # You can choose any colormap you like
    marker='o'
)
# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Fruit Number')  # Label for the colorbar
# Add labels and adjust layout
plt.xlabel(r'Leaf area [m$^2$]')
plt.ylabel(r'Fruit occlusion ratio')
plt.gca().set_box_aspect(1)
plt.show()


#%%
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import plotly.graph_objects as go

df_Leaf_Fruits = pd.read_csv('df_Leaf_Fruits.csv')

# Modify the equation to match curve_fit requirements
# First argument should be independent variables (x, y), followed by parameters to fit (k, l, P)
def equation(xy, k, l, P): 
    x, y = xy
    return (k*(P-y)+l)*x/((k*(P-y)+l)-x)

# Prepare the data
xdata = np.vstack((df_result['fruit_count_real'], 
                   df_result['leaf_pixels_million_real']))
ydata = df_result['fruit_number']

# Initial parameter guesses
p0 = [1.0, 1.0, 1.0]  # Initial guesses for k, l, P

# Fit the equation
popt, pcov = curve_fit(equation, xdata, ydata, p0=p0)

# Print the fitted parameters
print("Fitted parameters (k, l, P):", popt)

# add the surface of the predicted result
# Create a grid of x and y values
x_range_sim = np.linspace(df_result['fruit_count_real'].min(), 
                      df_result['fruit_count_real'].max(), 
                      200)
y_range_sim = np.linspace(df_result['leaf_pixels_million_real'].min(), 
                      6, 
                      200)
X_sim, Y_sim = np.meshgrid(x_range_sim, y_range_sim)
# Calculate Z values for the surface
Z_sim = equation(np.vstack([X_sim.ravel(), Y_sim.ravel()]), *popt).reshape(X_sim.shape)

# # Create a mask for the regions you want to exclude
# # # Example 1: Exclude points where Z values are above a threshold
# # mask = Z > some_threshold
# # OR
# # Example 2: Exclude points based on X and Y conditions
# mask = (X_sim > 105) & (Y_sim > 4.9)
# # OR
# # # Example 3: Exclude points outside the data range
# # mask = (Z < df_result_leafpixels['fruit_number_50cmH'].min()) | (Z > df_result_leafpixels['fruit_number_50cmH'].max())

# # Apply the mask
# Z_sim[mask] = np.nan

# Create the 3D plot
fig = go.Figure()

# # Add scatter points for simulation data
fig.add_trace(go.Scatter3d(
    x=df_result['fruit_count_real'],
    y=df_result['leaf_pixels_million_real'],
    z=df_result['fruit_number'],
    mode='markers',
    marker=dict(size=5),
    name='Simulation Data'
))

# Add the surface
fig.add_trace(go.Surface(
    x=x_range_sim,
    y=y_range_sim,
    z=Z_sim,
    opacity=0.7,
    name='Fitted Surface'
))

# Map the Date column to colors
# Define a color map for the dates
date_color_map = {
    '2022-12-28': 'yellow',
    '2023-01-30': 'orange',
    '2023-01-31': 'red'
}
colors = df_Leaf_Fruits['Date'].map(date_color_map)

fig.add_trace(go.Scatter3d(
    x=df_Leaf_Fruits['FruitNumDetected'],
    y=df_Leaf_Fruits['Leaf_pixels_million'],
    z=df_Leaf_Fruits['FruitNumOnPlant'],
    mode='markers',
    marker=dict(
        size=5, 
        # color = 'red',
        color = colors,
        opacity = 0.7
    ),
    name='Actual Data'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Detected number of fruits [-]',
        yaxis_title='Counted pixels of leaves [10⁶pixels]',
        zaxis_title='Number of fruits on plants [-]', 
        zaxis_range=[0, 250]
    ),
    width=500,
    height=500
)

fig.show()

fig.write_html('3d_detfruit_leafpixel_plantfruit.html')


#%%
# Liner regression analysis of leaf pixels vs leaf area

# def equation_leaf(x, k, l): 
#     return k * x**2 + l * x

import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import plotly.graph_objects as go

saturation_pixel = 11.22

def equation_leaf(x, b, c): 
    return (saturation_pixel * b * x / (saturation_pixel - x)) ** c

# Prepare the data
xdata = df_result['leaf_pixels_million_real']
ydata = df_result['leaf_area']

# Initial parameter guesses
p0_leaf = [1.0, 1.0]  # Initial guesses for k, l

# Fit the equation
popt_leaf, pcov_leaf = curve_fit(equation_leaf, xdata, ydata, p0=p0_leaf)

# Print the fitted parameters
print("Fitted parameters (b, c):", popt_leaf)

x_seq = np.linspace(xdata.min(), xdata.max(), 100)

# plot the data and the fitted curve
plt.figure(figsize=(2.4, 2.4), dpi=200)
plt.plot(x_seq, equation_leaf(x_seq, *popt_leaf), color='gray', label='Fitted curve', zorder=1)
plt.scatter(xdata, ydata, color='black', label='Data', zorder=2)
plt.xlabel(r'Leaf pixels [10$^6$pixels]')
plt.ylabel(r'Leaf area [m$^2$]')
plt.xlim(left=0, right=6)
plt.ylim(bottom=0, top=1.6)
plt.xticks([0, 1, 2, 3, 4, 5, 6])
plt.yticks([0, 0.5, 1, 1.5])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
plt.show()


#%%
# Actual vs Estimation about fruit number
df_Leaf_Fruits['estimatedFruitNumOnPlant'] = equation_expand(np.vstack([df_Leaf_Fruits['FruitNumDetected'], df_Leaf_Fruits['Leaf_pixels_million']]), *popt_expand)
df_Leaf_Fruits['estminusactFruitNumOnPlant'] = df_Leaf_Fruits['estimatedFruitNumOnPlant'] - df_Leaf_Fruits['FruitNumOnPlant']
df_Leaf_Fruits['estimatedLeafArea'] = equation_leaf(df_Leaf_Fruits['Leaf_pixels_million'], *popt_leaf)
df_Leaf_Fruits['estminusactLeafArea'] = df_Leaf_Fruits['estimatedLeafArea'] - df_Leaf_Fruits['LeafArea_m2']

# RMSE and RMSE persent of FruitNumOnPlant
RMSE = np.sqrt(np.mean(df_Leaf_Fruits['estminusactFruitNumOnPlant']**2))
RMSE_percent = RMSE / df_Leaf_Fruits['FruitNumOnPlant'].mean() * 100
print(f'RMSE for FruitNumOnPlant: {RMSE:.2f}')
print(f'RMSE percent for FruitNumOnPlant: {RMSE_percent:.2f}%')

# RMSE and RMSE persent of LeafArea
RMSE_leaf = np.sqrt(np.mean(df_Leaf_Fruits['estminusactLeafArea']**2))
RMSE_percent_leaf = RMSE_leaf / df_Leaf_Fruits['LeafArea_m2'].mean() * 100
print(f'RMSE for LeafArea: {RMSE_leaf:.2f}')
print(f'RMSE percent for LeafArea: {RMSE_percent_leaf:.2f}%')

# RMSE and RMSE persent of LeafArea only 2023-01-30 and 2023-01-31
df_Leaf_Fruits_0130and0131 = df_Leaf_Fruits[df_Leaf_Fruits['Date'].isin(['2023-01-30', '2023-01-31'])]
RMSE_leaf_0130and0131 = np.sqrt(np.mean(df_Leaf_Fruits_0130and0131['estminusactLeafArea']**2))
RMSE_percent_leaf_0130and0131 = RMSE_leaf_0130and0131 / df_Leaf_Fruits_0130and0131['LeafArea_m2'].mean() * 100
print(f'RMSE for LeafArea on 2023-01-30 and 2023-01-31: {RMSE_leaf_0130and0131:.2f}')
print(f'RMSE percent for LeafArea on 2023-01-30 and 2023-01-31: {RMSE_percent_leaf_0130and0131:.2f}%')


# Make 2d plot of FruitNumOnPlant vs estminusactFruitNumOnPlant
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# plot of leaf area vs estimated leaf area
plt.figure(figsize=(2.4, 2.4), dpi=200)
# Add a line of 1:1
plt.plot(np.linspace(0, 2.7, 100), np.linspace(0, 2.7, 100), color='gray', linestyle='--')
# Define a colormap
cmap = cm.get_cmap('viridis')  # You can choose any colormap you like
# Normalize the FruitNumOnPlant values to the range [0, 1] for the colormap
norm = plt.Normalize(df_Leaf_Fruits['Leaf_pixels_million'].min(), df_Leaf_Fruits['Leaf_pixels_million'].max())
dates_to_plot = df_Leaf_Fruits['Date'].unique()
markers = ['o', '^', 's']
# color of markers differes depending on the value of FruitNumOnPlant
for i, date in enumerate(dates_to_plot):
    group = df_Leaf_Fruits[df_Leaf_Fruits['Date'] == date]
    plt.scatter(
        group['LeafArea_m2'], 
        group['estimatedLeafArea'], 
        marker=markers[i], 
        color=cmap(norm(group['Leaf_pixels_million'])),
        s=15, 
        label=f'Data Points {date}', 
        zorder=2  # Ensure scatter points are on top)
    )
plt.xlabel(r'Leaf area [m$^2$]')
plt.ylabel(r'Estimated leaf area [m$^2$]')
# Set axes to start from zero
plt.xlim(left=0, right=2.0)
plt.ylim(bottom=0, top=2.0)
# Set x axis tick: 0, 50, 100, 150, 200, 250
plt.xticks([0, 0.5, 1, 1.5, 2])
# Set y axis tick: 0, 50, 100, 150, 200, 250
plt.yticks([0, 0.5, 1, 1.5, 2])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
# Show color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label(r'Leaf pixels [10$^6$pixels]')
# font size of x and y labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# plot of FruitNumOnPlant vs estimated FruitNumOnPlant
plt.figure(figsize=(2.4, 2.4), dpi=200)
# Add a line of 1:1
plt.plot(np.linspace(0, 250, 100), np.linspace(0, 250, 100), color='gray', linestyle='--')
# Define a colormap
cmap = cm.get_cmap('viridis')  # You can choose any colormap you like
# Normalize the FruitNumOnPlant values to the range [0, 1] for the colormap
norm = plt.Normalize(df_Leaf_Fruits['Leaf_pixels_million'].min(), df_Leaf_Fruits['Leaf_pixels_million'].max())
dates_to_plot = df_Leaf_Fruits['Date'].unique()
markers = ['o', '^', 's']
# color of markers differes depending on the value of FruitNumOnPlant
for i, date in enumerate(dates_to_plot):
    group = df_Leaf_Fruits[df_Leaf_Fruits['Date'] == date]
    plt.scatter(
        group['FruitNumOnPlant'], 
        group['estimatedFruitNumOnPlant'], 
        marker=markers[i], 
        # color=cmap(norm(group['LeafArea_m2'])),
        color=cmap(norm(group['Leaf_pixels_million'])),
        s=15, 
        label=f'Data Points {date}', 
        zorder=2  # Ensure scatter points are on top)
    )
plt.xlabel(r'Fruit number on plants [-]')
plt.ylabel(r'Estimated number of fruits on plants [-]')
# Set axes to start from zero
plt.xlim(left=0, right=250)
plt.ylim(bottom=0, top=250)
# Set x axis tick: 0, 50, 100, 150, 200, 250
plt.xticks([0, 50, 100, 150, 200, 250])
# Set y axis tick: 0, 50, 100, 150, 200, 250
plt.yticks([0, 50, 100, 150, 200, 250])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
# Show color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
# cbar.set_label(r'Leaf area [m$^2$]')
cbar.set_label(r'Leaf pixels [10$^6$pixels]')
# font size of x and y labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# Make 3d plot of x: FruitNumDetected, y: Leaf_pixels_million, z: estminusactFruitNumOnPlant
import plotly.express as px

# Make 3D plot of x: FruitNumDetected, y: Leaf_pixels_million, z: estminusactFruitNumOnPlant
fig = px.scatter_3d(
    df_Leaf_Fruits, 
    x='FruitNumDetected', 
    y='Leaf_pixels_million', 
    z='estminusactFruitNumOnPlant', 
    # color='Date'  # Color by Date
)

# Map the Date column to colors
# Define a color map for the dates
date_color_map = {
    '2022-12-28': 'yellow',
    '2023-01-30': 'orange',
    '2023-01-31': 'red'
}
colors = df_Leaf_Fruits['Date'].map(date_color_map)
# Update marker properties if needed
fig.update_traces(
    marker=dict(
        size=5, 
        symbol='circle',
        color=colors
    )
)

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Detected number of fruits [-]',
        yaxis_title='Counted pixels of leaves [10⁶pixels]',
        zaxis_title='Estimated - Actual number of fruits on plants [-]', 
    )
)

fig.show()

#%%
# calculate the mean of FruitNumDetected, Leaf_pixels_million, FruitNumOnPlant for each Date
df_Leaf_Fruits_mean = df_Leaf_Fruits.groupby('Date').mean()
df_Leaf_Fruits_mean['NewEstimatedFruitNumOnPlant'] = equation(np.vstack([df_Leaf_Fruits_mean['FruitNumDetected'], df_Leaf_Fruits_mean['Leaf_pixels_million']]), *popt)
df_Leaf_Fruits_mean['NewEstminusactFruitNumOnPlant'] = df_Leaf_Fruits_mean['NewEstimatedFruitNumOnPlant'] - df_Leaf_Fruits_mean['FruitNumOnPlant']

# Make 3d plot of x: FruitNumDetected, y: Leaf_pixels_million, z: FruitNumOnPlant
fig = px.scatter_3d(
    df_Leaf_Fruits_mean, 
    x='FruitNumDetected', 
    y='Leaf_pixels_million', 
    z='FruitNumOnPlant', 
)
fig.show()


#%%
# Regression analysis
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
df_Leaf_Fruits_std = df_Leaf_Fruits.copy()
df_Leaf_Fruits_std['interaction_term'] = np.sqrt(df_Leaf_Fruits_std['FruitNumDetected'] * df_Leaf_Fruits_std['Leaf_pixels_million'])
df_Leaf_Fruits_std[['FruitNumDetected', 'Leaf_pixels_million', 'interaction_term']] = scaler_X.fit_transform(df_Leaf_Fruits_std[['FruitNumDetected', 'Leaf_pixels_million', 'interaction_term']])
df_Leaf_Fruits_std['FruitNumOnPlant'] = scaler_y.fit_transform(df_Leaf_Fruits_std[['FruitNumOnPlant']])

# Prepare the independent variables (X)
X = sm.add_constant(np.column_stack([
    df_Leaf_Fruits_std['FruitNumDetected'],
    df_Leaf_Fruits_std['Leaf_pixels_million'],
    df_Leaf_Fruits_std['interaction_term']
]))

# Prepare the dependent variable (y)
y = df_Leaf_Fruits_std['FruitNumOnPlant']

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Get predictions from standardized data
y_pred_std = model.predict(X)

# Transform predictions back to original scale
y_pred = y_pred_std * scaler_y.scale_[0] + scaler_y.mean_[0]  # 0 index for FruitNumOnPlant

# Plot actual vs predicted values
plt.figure(figsize=(4, 4), dpi=200)
plt.scatter(df_Leaf_Fruits['FruitNumOnPlant'], y_pred)
plt.plot([0, df_Leaf_Fruits['FruitNumOnPlant'].max()], 
         [0, df_Leaf_Fruits['FruitNumOnPlant'].max()], 
         'r--', label='1:1 line')
plt.xlabel('Actual FruitNumOnPlant')
plt.ylabel('Predicted FruitNumOnPlant')
plt.gca().set_aspect('equal')
plt.show()

# Calculate R-squared
r2 = model.rsquared
print(f"R-squared: {r2:.4f}")

# Calculate RMSE (Root Mean Square Error)
rmse = np.sqrt(np.mean((df_Leaf_Fruits['FruitNumOnPlant'] - y_pred)**2))
print(f"RMSE: {rmse:.4f}")

#%%
import plotly.graph_objects as go
import numpy as np

# Create a grid of points for FruitNumDetected and Leaf_pixels_million
x_range = np.linspace(df_Leaf_Fruits['FruitNumDetected'].min(), 
                      df_Leaf_Fruits['FruitNumDetected'].max(), 
                      50)
y_range = np.linspace(df_Leaf_Fruits['Leaf_pixels_million'].min(), 
                      df_Leaf_Fruits['Leaf_pixels_million'].max(), 
                      50)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Create grid data for prediction
grid_data = pd.DataFrame({
    'FruitNumDetected': X_grid.flatten(),
    'Leaf_pixels_million': Y_grid.flatten()
})
grid_data['interaction_term'] = np.sqrt(grid_data['FruitNumDetected'] * grid_data['Leaf_pixels_million'])

# Standardize grid data using the same scaler
grid_data_std = grid_data.copy()
grid_data_std[['FruitNumDetected', 'Leaf_pixels_million', 'interaction_term']] = scaler_X.transform(
    grid_data[['FruitNumDetected', 'Leaf_pixels_million', 'interaction_term']]
)

# Prepare X matrix for prediction
X_pred = sm.add_constant(np.column_stack([
    grid_data_std['FruitNumDetected'],
    grid_data_std['Leaf_pixels_million'],
    grid_data_std['interaction_term']
]))

# Make predictions and transform back to original scale
Z_pred_std = model.predict(X_pred)
Z_pred = Z_pred_std * scaler_y.scale_[0] + scaler_y.mean_[0]
Z_grid = Z_pred.reshape(X_grid.shape)

# Create 3D plot
fig = go.Figure()

# Map the Date column to colors
# Define a color map for the dates
date_color_map = {
    '2022-12-28': 'yellow',
    '2023-01-30': 'orange',
    '2023-01-31': 'red'
}
colors = df_Leaf_Fruits['Date'].map(date_color_map)

# Add scatter points for actual data
fig.add_trace(go.Scatter3d(
    x=df_Leaf_Fruits['FruitNumDetected'],
    y=df_Leaf_Fruits['Leaf_pixels_million'],
    z=df_Leaf_Fruits['FruitNumOnPlant'],
    mode='markers',
    marker=dict(size=5,
        color = colors,
        opacity = 0.7
    ),
    name='Actual Data'
))

# Add surface for model predictions
fig.add_trace(go.Surface(
    x=x_range,
    y=y_range,
    z=Z_grid,
    opacity=0.7,
    colorscale='viridis',
    name='Model Surface'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Detected number of fruits [-]',
        yaxis_title='Counted pixels of leaves [10⁶pixels]',
        zaxis_title='Number of fruits on plants [-]'
    ),
    width=500,
    height=500,
    title='Model Surface with Actual Data Points'
)


fig.show()

# Save the figure if needed
# fig.write_html("model_surface_3d.html")

#%%

# Create the 3D plot
fig = go.Figure()

# Map the Date column to colors
# Define a color map for the dates
date_color_map = {
    '2022-12-28': 'yellow',
    '2023-01-30': 'orange',
    '2023-01-31': 'red'
}
colors = df_Leaf_Fruits['Date'].map(date_color_map)

# Add the surface
fig.add_trace(go.Surface(
    x=x_range_sim,
    y=y_range_sim,
    z=Z_sim,
    opacity=0.7,
    name='Fitted Surface of simulation data'
))

# Add surface for model predictions
fig.add_trace(go.Surface(
    x=x_range,
    y=y_range,
    z=Z_grid,
    opacity=0.7,
    colorscale='viridis',
    name='Fitted surface of actual data'
))

# Add scatter points for simulation data
fig.add_trace(go.Scatter3d(
    x=df_result['fruit_count_real'],
    y=df_result['leaf_pixels_million_real'],
    z=df_result['fruit_number'],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Simulation Data'
))


fig.add_trace(go.Scatter3d(
    x=df_Leaf_Fruits['FruitNumDetected'],
    y=df_Leaf_Fruits['Leaf_pixels_million'],
    z=df_Leaf_Fruits['FruitNumOnPlant'],
    mode='markers',
    marker=dict(
        size=5, 
        # color = 'red',
        color = colors,
        opacity = 0.7
    ),
    name='Actual Data'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Detected number of fruits [-]',
        yaxis_title='Counted pixels of leaves [10⁶pixels]',
        zaxis_title='Number of fruits on plants [-]', 
        zaxis_range=[0, 250]
    ),
    width=500,
    height=500
)

fig.show()


#%%

# Fit the equation to the data
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import plotly.graph_objects as go

df_Leaf_Fruits = pd.read_csv('df_Leaf_Fruits.csv')

# Modify the equation to match curve_fit requirements
# First argument should be independent variables (x, y), followed by parameters to fit (k, l, P)
def equation(xy, k, l, P): 
    x, y = xy
    return (k*(P-y)+l)*x/((k*(P-y)+l)-x)

# Prepare the data
xdata = np.vstack((df_Leaf_Fruits['FruitNumDetected'], 
                   df_Leaf_Fruits['Leaf_pixels_million']))
ydata = df_Leaf_Fruits['FruitNumOnPlant']

# Initial parameter guesses
p0 = [1.0, 1.0, 1.0]  # Initial guesses for k, l, P

# Fit the equation
popt, pcov = curve_fit(equation, xdata, ydata, p0=p0)

# summary of the curve fit
print(popt)

# Print the fitted parameters
print("Fitted parameters (k, l, P):", popt)

# Create a grid of points for FruitNumDetected and Leaf_pixels_million
x_range = np.linspace(df_Leaf_Fruits['FruitNumDetected'].min(), 
                      df_Leaf_Fruits['FruitNumDetected'].max(), 
                      50)
y_range = np.linspace(df_Leaf_Fruits['Leaf_pixels_million'].min(), 
                      df_Leaf_Fruits['Leaf_pixels_million'].max(), 
                      50)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Create grid data for prediction
grid_data = pd.DataFrame({
    'FruitNumDetected': X_grid.flatten(),
    'Leaf_pixels_million': Y_grid.flatten()
})
Z_grid = equation(np.vstack([grid_data['FruitNumDetected'], grid_data['Leaf_pixels_million']]), *popt)
Z_grid = Z_grid.reshape(X_grid.shape)

# Create 3D plot
fig = go.Figure()

# Map the Date column to colors
# Define a color map for the dates
date_color_map = {
    '2022-12-28': 'yellow',
    '2023-01-30': 'orange',
    '2023-01-31': 'red'
}
colors = df_Leaf_Fruits['Date'].map(date_color_map)

# add surface for the model
fig.add_trace(go.Surface(
    x=x_range,
    y=y_range,
    z=Z_grid,
    opacity=0.7,
    colorscale='viridis',
    name='Model Surface'
))

# Add scatter points for actual data
fig.add_trace(go.Scatter3d(
    x=df_Leaf_Fruits['FruitNumDetected'],
    y=df_Leaf_Fruits['Leaf_pixels_million'],
    z=df_Leaf_Fruits['FruitNumOnPlant'],
    mode='markers',
    marker=dict(size=5,
        color = colors,
        opacity = 0.7
    ),
    name='Actual Data'
))

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title='Detected number of fruits [-]',
        yaxis_title='Counted pixels of leaves [10⁶pixels]',
        zaxis_title='Number of fruits on plants [-]',
        # xaxis=dict(range=[0, 100]),  # Set x-axis range
        # yaxis=dict(range=[0, 10]),   # Set y-axis range
        zaxis=dict(range=[0, 200])   # Set z-axis range
    ),
    width=500,
    height=500,
    title='Model Surface with Actual Data Points'
)

fig.show()

# Save the figure if needed
# fig.write_html("model_surface_3d.html")

#%%
# Another version of equation (expand expression)
# Fit the equation to the data
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import plotly.graph_objects as go

# Modify the equation to match curve_fit requirements
# First argument should be independent variables (x, y), followed by parameters to fit (k, l, P)
def equation_expand(xy, k, l): 
    x, y = xy
    return (-k*y+l)*x/((-k*y+l)-x)

# Prepare the data
xdata = np.vstack((df_result['fruit_count_real'], 
                   df_result['leaf_pixels_million_real']))
ydata = df_result['fruit_number']

# Initial parameter guesses
p0_expand = [50, 500]  # Initial guesses for k, l

# Fit the equation
popt_expand, pcov_expand = curve_fit(equation_expand, xdata, ydata, p0=p0_expand)

# summary of the curve fit
print(popt_expand)

# Print the fitted parameters
print("Fitted parameters (k, l):", popt_expand)



#%%
import scipy.stats as stats
import matplotlib.cm as cm

# Make figure of Leaf area vs Leaf pixels for each Date
cmap = cm.get_cmap('viridis')  # You can choose any colormap you like
# Normalize the FruitNumOnPlant values to the range [0, 1] for the colormap
norm = plt.Normalize(df_Leaf_Fruits['FruitNumOnPlant'].min(), df_Leaf_Fruits['FruitNumOnPlant'].max())
dates_to_plot = df_Leaf_Fruits['Date'].unique()
markers = ['o', '^', 's']
plt.figure(figsize=(3, 3), dpi=200)
for i, date in enumerate(dates_to_plot):
    group = df_Leaf_Fruits[df_Leaf_Fruits['Date'] == date]
    plt.scatter(
        group['LeafArea_m2'], 
        group['Leaf_pixels_million'], 
        marker=markers[i], 
        color=cmap(norm(group['FruitNumOnPlant'])),
        s=15, 
        label=f'Data Points {date}'
    )
plt.xlabel(r'Leaf area [m$^2$]')
plt.ylabel(r'Leaf pixels [10$^6$pixels]')
# Set axes to start from zero
plt.xlim(left=0, right=1.5)
plt.ylim(bottom=0, top=6.5)
# Set x axis tick: 0, 50, 100, 150, 200, 250
plt.xticks([0, 0.5, 1, 1.5])
# Set y axis tick: 0, 50, 100, 150, 200, 250
plt.yticks([0, 2, 4, 6])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
# Show color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label(r'Fruit number on plants [-]')
# font size of x and y labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# correlation coefficient between Leaf area and Leaf pixels
print(f'Correlation coefficient between Leaf area and Leaf pixels: {np.corrcoef(df_Leaf_Fruits["LeafArea_m2"], df_Leaf_Fruits["Leaf_pixels_million"])[0, 1]}')

# Test of significance of correlation coefficient between Leaf area and Leaf pixels
print(f'p-value of correlation coefficient between Leaf area and Leaf pixels: {stats.ttest_rel(df_Leaf_Fruits["LeafArea_m2"], df_Leaf_Fruits["Leaf_pixels_million"])[1]}')


# Make figure of FruitNumDetected vs FruitNumOnPlant for each Date
cmap = cm.get_cmap('viridis')  # You can choose any colormap you like
# Normalize the FruitNumOnPlant values to the range [0, 1] for the colormap
norm = plt.Normalize(df_Leaf_Fruits['LeafArea_m2'].min(), df_Leaf_Fruits['LeafArea_m2'].max())
dates_to_plot = df_Leaf_Fruits['Date'].unique()
markers = ['o', '^', 's']
plt.figure(figsize=(3, 3), dpi=200)
for i, date in enumerate(dates_to_plot):
    group = df_Leaf_Fruits[df_Leaf_Fruits['Date'] == date]
    plt.scatter(
        group['FruitNumOnPlant'], 
        group['FruitNumDetected'], 
        marker=markers[i], 
        color=cmap(norm(group['LeafArea_m2'])),
        s=15, 
        label=f'Data Points {date}'
    )
plt.xlabel(r'Fruit number on plants')
plt.ylabel(r'Fruit number detected')
# Set axes to start from zero
plt.xlim(left=0, right=270)
plt.ylim(bottom=0, top=270)
# Add y = x line as dashed line
plt.plot([0, 270], [0, 270], color='gray', linestyle='--', label='y = x')
# Set x axis tick: 0, 50, 100, 150, 200, 250
plt.xticks([0, 50, 100, 150, 200, 250])
# Set y axis tick: 0, 50, 100, 150, 200, 250
plt.yticks([0, 50, 100, 150, 200, 250])
# Ensure the plot area is square
plt.gca().set_box_aspect(1)
# Show color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label(r'Leaf area [m$^2$]')
# font size of x and y labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# correlation coefficient between FruitNumOnPlant and FruitNumDetected
print(f'Correlation coefficient between FruitNumOnPlant and FruitNumDetected: {np.corrcoef(df_Leaf_Fruits["FruitNumOnPlant"], df_Leaf_Fruits["FruitNumDetected"])[0, 1]}')

# Test of significance of correlation coefficient between FruitNumOnPlant and FruitNumDetected
print(f'p-value of correlation coefficient between FruitNumOnPlant and FruitNumDetected: {stats.ttest_rel(df_Leaf_Fruits["FruitNumOnPlant"], df_Leaf_Fruits["FruitNumDetected"])[1]}')
