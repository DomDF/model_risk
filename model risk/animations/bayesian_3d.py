#!/usr/bin/env python

from manim import *
import numpy as np
from scipy import stats

# Define a constant for the font
FONT = "Atkinson Hyperlegible"

class BayesianUpdate3D(ThreeDScene):
    def construct(self):
        # Set the camera orientation
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES, zoom=0.7)
        self.camera.background_color = "#111111"
        
        # Create the 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 0.2, 0.05],
            x_length=8,
            y_length=8,
            z_length=4,
            axis_config={"include_tip": False, "include_numbers": False, "include_ticks": False}
        )
        
        # Initial parameters for the prior (2D Gaussian)
        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.array([[2.0, 0.5], [0.5, 2.0]])  # Covariance with some correlation
        
        # Function to create a 2D Gaussian surface
        def get_gaussian_surface(mean, cov, resolution=50):
            # Create a grid of points
            x = np.linspace(-4, 4, resolution)
            y = np.linspace(-4, 4, resolution)
            X, Y = np.meshgrid(x, y)
            
            # Flatten the grid for multivariate normal calculation
            pos = np.dstack((X, Y))
            
            # Calculate the PDF values
            rv = stats.multivariate_normal(mean, cov)
            Z = rv.pdf(pos)
            
            # Create the surface
            surface = Surface(
                lambda u, v: axes.c2p(
                    x[int(u * (resolution-1))],
                    y[int(v * (resolution-1))],
                    Z[int(v * (resolution-1)), int(u * (resolution-1))]
                ),
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(resolution, resolution),
                fill_opacity=0.8
            )
            
            return surface, Z.max()
        
        # Create the prior surface
        prior_surface, max_height = get_gaussian_surface(prior_mean, prior_cov)
        prior_surface.set_color(BLUE)
        
        # Add a subtle grid on the xy-plane
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        grid.set_opacity(0.2)
        
        # Add the axes and surface
        self.add(grid)
        self.play(
            Create(prior_surface),
            run_time=2
        )
        self.wait(1)
        
        # Add a title that appears above the scene
        title = Text("Bayesian Updating in Action", font=FONT, font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # True parameter value (where data will be generated from)
        true_mean = np.array([1.5, 1.0])
        true_cov = np.array([[0.5, 0.0], [0.0, 0.5]])  # Smaller, uncorrelated covariance
        
        # Generate data points from the true distribution
        np.random.seed(42)  # For reproducibility
        num_points = 8
        data_points = np.random.multivariate_normal(true_mean, true_cov, size=num_points)
        
        # Current parameters (starting with prior)
        current_mean = prior_mean.copy()
        current_cov = prior_cov.copy()
        current_surface = prior_surface
        
        # For each data point, update the distribution
        for i, data_point in enumerate(data_points):
            # Create a visual representation of the data point
            data_sphere = Sphere(radius=0.1, fill_color=YELLOW)
            data_sphere.move_to(axes.c2p(data_point[0], data_point[1], max_height * 1.5))
            
            # Add some "observational noise" particles around the data point
            noise_particles = VGroup()
            for _ in range(10):
                noise = np.random.normal(0, 0.2, 2)
                noise_point = data_point + noise
                particle = Dot3D(axes.c2p(noise_point[0], noise_point[1], max_height * 1.5), 
                                radius=0.02, color=YELLOW_A)
                noise_particles.add(particle)
            
            # Show the data point falling from above
            self.play(
                FadeIn(data_sphere),
                FadeIn(noise_particles)
            )
            
            # Data point falls onto the surface
            target_z = axes.c2p(0, 0, 0)[2]  # z-coordinate of the xy-plane
            rv = stats.multivariate_normal(current_mean, current_cov)
            pdf_height = rv.pdf(data_point)
            target_position = axes.c2p(data_point[0], data_point[1], pdf_height)
            
            self.play(
                data_sphere.animate.move_to(target_position),
                noise_particles.animate.move_to(target_position),
                run_time=1.5
            )
            
            # Data point is "absorbed" into the surface
            self.play(
                data_sphere.animate.scale(0.5).set_opacity(0),
                noise_particles.animate.set_opacity(0),
                run_time=1
            )
            
            # Calculate the posterior parameters (using Bayesian update for multivariate normal)
            # Precision matrices (inverse of covariance)
            prior_precision = np.linalg.inv(current_cov)
            # Likelihood precision (assuming known covariance of 1.0 in each dimension)
            likelihood_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
            likelihood_precision = np.linalg.inv(likelihood_cov)
            
            # Posterior precision is the sum of prior and likelihood precisions
            posterior_precision = prior_precision + likelihood_precision
            posterior_cov = np.linalg.inv(posterior_precision)
            
            # Posterior mean
            posterior_mean = posterior_cov @ (prior_precision @ current_mean + likelihood_precision @ data_point)
            
            # Create the posterior surface
            posterior_surface, _ = get_gaussian_surface(posterior_mean, posterior_cov)
            posterior_surface.set_color(RED)
            
            # Morph the prior surface into the posterior
            self.play(
                Transform(current_surface, posterior_surface),
                run_time=2
            )
            self.wait(0.5)
            
            # Update for next iteration
            current_mean = posterior_mean
            current_cov = posterior_cov
            
            # The posterior becomes the new prior (change color back to blue)
            new_prior_surface, _ = get_gaussian_surface(current_mean, current_cov)
            new_prior_surface.set_color(BLUE)
            
            self.play(
                Transform(current_surface, new_prior_surface),
                run_time=1
            )
            self.wait(0.5)
        
        # Show the final distribution and the true distribution
        true_surface, _ = get_gaussian_surface(true_mean, true_cov)
        true_surface.set_color(GREEN)
        
        # Add a subtle pulse to highlight the true distribution
        self.play(
            FadeIn(true_surface),
            run_time=1.5
        )
        
        # Pulse the true distribution
        self.play(
            true_surface.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=1
        )
        
        # Show both distributions
        self.wait(1)
        
        # Rotate the camera to show the distributions from different angles
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # Fade everything out
        self.play(
            FadeOut(current_surface),
            FadeOut(true_surface),
            FadeOut(grid),
            FadeOut(title),
            run_time=1.5
        )
        self.wait(1)


class BayesianConcept3D(ThreeDScene):
    def construct(self):
        # Set the camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES, zoom=0.7)
        self.camera.background_color = "#111111"
        
        # Create the 3D axes (but don't display them)
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[0, 0.2, 0.05],
            x_length=8,
            y_length=8,
            z_length=4,
        )
        
        # Add a subtle grid on the xy-plane
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": GREY_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        grid.set_opacity(0.2)
        self.add(grid)
        
        # Add a title that appears above the scene
        title = Text("From Prior Belief to Posterior Knowledge", font=FONT, font_size=36)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        
        # Function to create a 2D Gaussian surface
        def get_gaussian_surface(mean, cov, resolution=50, color=BLUE):
            # Create a grid of points
            x = np.linspace(-4, 4, resolution)
            y = np.linspace(-4, 4, resolution)
            X, Y = np.meshgrid(x, y)
            
            # Flatten the grid for multivariate normal calculation
            pos = np.dstack((X, Y))
            
            # Calculate the PDF values
            rv = stats.multivariate_normal(mean, cov)
            Z = rv.pdf(pos)
            
            # Create the surface
            surface = Surface(
                lambda u, v: axes.c2p(
                    x[int(u * (resolution-1))],
                    y[int(v * (resolution-1))],
                    Z[int(v * (resolution-1)), int(u * (resolution-1))]
                ),
                u_range=[0, 1],
                v_range=[0, 1],
                resolution=(resolution, resolution),
                fill_opacity=0.8
            )
            
            # Apply a color gradient
            surface.set_color(color)
            
            return surface, Z.max()
        
        # Initial parameters for the prior (2D Gaussian)
        prior_mean = np.array([0.0, 0.0])
        prior_cov = np.array([[2.0, 0.5], [0.5, 2.0]])  # Covariance with some correlation
        
        # Create the prior surface
        prior_surface, max_height = get_gaussian_surface(prior_mean, prior_cov, color=BLUE_C)
        
        # Add a label for the prior
        prior_label = Text("Prior", font=FONT, font_size=24, color=BLUE_C)
        prior_label.to_edge(LEFT).to_edge(UP, buff=1.5)
        self.add_fixed_in_frame_mobjects(prior_label)
        
        # Show the prior surface
        self.play(
            Create(prior_surface),
            Write(prior_label),
            run_time=2
        )
        self.wait(1)
        
        # True parameter value (where data will be generated from)
        true_mean = np.array([2.0, 1.0])
        true_cov = np.array([[0.5, 0.0], [0.0, 0.5]])  # Smaller, uncorrelated covariance
        
        # Generate data points from the true distribution
        np.random.seed(42)  # For reproducibility
        num_points = 10
        data_points = np.random.multivariate_normal(true_mean, true_cov, size=num_points)
        
        # Create a visual representation of all data points
        data_group = VGroup()
        for point in data_points:
            data_sphere = Sphere(radius=0.1, fill_color=YELLOW)
            data_sphere.move_to(axes.c2p(point[0], point[1], max_height * 1.5))
            data_group.add(data_sphere)
        
        # Add a label for the data
        data_label = Text("New Data", font=FONT, font_size=24, color=YELLOW)
        data_label.next_to(prior_label, DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(data_label)
        
        # Show all data points appearing
        self.play(
            FadeIn(data_group),
            Write(data_label),
            run_time=1.5
        )
        self.wait(1)
        
        # Create a "rain" effect of data points falling onto the surface
        animations = []
        for i, sphere in enumerate(data_group):
            point = data_points[i]
            rv = stats.multivariate_normal(prior_mean, prior_cov)
            pdf_height = rv.pdf(point)
            target_position = axes.c2p(point[0], point[1], pdf_height)
            
            # Add animation with slight delay based on index
            animations.append(
                Succession(
                    Wait(i * 0.1),  # Stagger the start times
                    sphere.animate.move_to(target_position).scale(0.8),
                )
            )
        
        # Play all animations together
        self.play(*animations, run_time=3)
        self.wait(1)
        
        # Data points are "absorbed" into the surface
        self.play(
            FadeOut(data_group),
            run_time=1.5
        )
        
        # Calculate the posterior parameters
        # Precision matrices (inverse of covariance)
        prior_precision = np.linalg.inv(prior_cov)
        
        # Likelihood precision (assuming known covariance)
        likelihood_cov = np.array([[0.5, 0.0], [0.0, 0.5]])
        likelihood_precision = np.linalg.inv(likelihood_cov)
        
        # Posterior precision is the sum of prior precision and n times likelihood precision
        posterior_precision = prior_precision + num_points * likelihood_precision
        posterior_cov = np.linalg.inv(posterior_precision)
        
        # Posterior mean (weighted average of prior mean and data mean)
        data_mean = np.mean(data_points, axis=0)
        posterior_mean = posterior_cov @ (prior_precision @ prior_mean + num_points * likelihood_precision @ data_mean)
        
        # Create the posterior surface
        posterior_surface, _ = get_gaussian_surface(posterior_mean, posterior_cov, color=RED_C)
        
        # Add a label for the posterior
        posterior_label = Text("Posterior", font=FONT, font_size=24, color=RED_C)
        posterior_label.next_to(data_label, DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(posterior_label)
        
        # Morph the prior surface into the posterior with a glow effect
        self.play(
            Transform(prior_surface, posterior_surface),
            Write(posterior_label),
            run_time=3
        )
        self.wait(1)
        
        # Create the true distribution surface
        true_surface, _ = get_gaussian_surface(true_mean, true_cov, color=GREEN_C)
        
        # Add a label for the true distribution
        true_label = Text("True Distribution", font=FONT, font_size=24, color=GREEN_C)
        true_label.next_to(posterior_label, DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(true_label)
        
        # Show the true distribution
        self.play(
            FadeIn(true_surface),
            Write(true_label),
            run_time=2
        )
        
        # Pulse the true distribution
        self.play(
            true_surface.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=1
        )
        self.wait(1)
        
        # Add an explanation
        explanation = Text(
            "As we observe more data, our posterior belief\napproaches the true distribution",
            font=FONT, font_size=24, color=WHITE
        )
        explanation.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(explanation)
        self.play(Write(explanation), run_time=2)
        
        # Rotate the camera to show the distributions from different angles
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        # Fade everything out
        self.play(
            FadeOut(prior_surface),
            FadeOut(true_surface),
            FadeOut(grid),
            FadeOut(title),
            FadeOut(prior_label),
            FadeOut(data_label),
            FadeOut(posterior_label),
            FadeOut(true_label),
            FadeOut(explanation),
            run_time=1.5
        )
        self.wait(1)


if __name__ == "__main__":
    print("This script contains two 3D Bayesian updating animations:")
    print("1. BayesianUpdate3D - Shows sequential updating with individual data points")
    print("2. BayesianConcept3D - Shows a more conceptual visualization of Bayesian updating")
    print("\nRun with: manim -pqm bayesian_3d.py BayesianUpdate3D")
    print("Or: manim -pqm bayesian_3d.py BayesianConcept3D") 