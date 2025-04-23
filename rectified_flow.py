import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RectifiedFlow:
    """
    Implements rectified flow dynamics for generative modeling.
    Rectified flow provides a direct path between noise and data distributions.
    """
    def __init__(self, sigma_min=0.002, sigma_max=80.0):
        """
        Initialize the rectified flow model.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def noise_schedule(self, t):
        """
        Noise schedule that determines the noise level at time t.
        
        Args:
            t: Time variable in [0, 1]
            
        Returns:
            Noise level sigma(t)
        """
        # Cosine noise schedule
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
    
    def drift_coefficient(self, t):
        """
        Calculate the drift coefficient for the rectified flow.
        
        Args:
            t: Time variable in [0, 1]
            
        Returns:
            Drift coefficient
        """
        return 1.0
    
    def sample_path(self, x_1, t, noise=None):
        """
        Sample from the path at time t.
        
        Args:
            x_1: Target data sample (B, C, H, W)
            t: Time variable in [0, 1] (B,)
            noise: Optional noise to use (B, C, H, W)
            
        Returns:
            Dictionary containing:
                x_t: Sample at time t
                velocity: Ground truth velocity
        """
        batch_size = x_1.shape[0]
        
        # Expand time dimension for broadcasting
        t_expanded = t.view(-1, *([1] * (len(x_1.shape) - 1)))
        
        # Get noise if not provided
        if noise is None:
            noise = torch.randn_like(x_1)
        
        # Interpolate between noise and data
        sigma_t = self.noise_schedule(t_expanded)
        interp_coef = t_expanded
        x_t = interp_coef * x_1 + (1 - interp_coef) * noise
        
        # Compute the ground truth velocity (dx_t/dt)
        velocity = x_1 - noise
        
        return {
            "x_t": x_t,
            "velocity": velocity,
            "noise": noise,
            "sigma_t": sigma_t
        }
    
    def loss_fn(self, model_output, target_velocity):
        """
        Compute the loss between predicted and target velocities.
        
        Args:
            model_output: Predicted velocity from the model
            target_velocity: Target velocity from the path
            
        Returns:
            Loss value
        """
        return torch.mean((model_output - target_velocity) ** 2)


class EulerSolver:
    """
    Implements Euler method for solving ODEs in the context of generative modeling.
    """
    def __init__(self, model, rectified_flow):
        """
        Initialize the Euler solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
        """
        self.model = model
        self.rf = rectified_flow
        
    def generate_sample(self, rna_expr, num_steps=100, device="cuda"):
        """
        Generate a sample by solving the ODE using Euler method.
        
        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Number of steps for the Euler method
            device: The device to run the computation on
            
        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]
        
        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size
        
        # Start from random noise (t=0)
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        # Create time steps (from t=0 to t=1)
        dt = 1.0 / num_steps
        times = torch.linspace(0, 1 - dt, num_steps, device=device)
        
        # Euler integration
        for i, t in enumerate(times):
            # Expand t for batch
            t_batch = torch.ones(batch_size, device=device) * t
            
            # Get velocity prediction from model
            with torch.no_grad():
                velocity = self.model(x, t_batch, rna_expr)
            
            # Update x using Euler method
            x = x + velocity * dt
            
            # Optional: Add progress logging for long generations
            if i % (num_steps // 10) == 0:
                logger.info(f"Generation progress: {i}/{num_steps} steps")
        
        # Clamp to valid image range
        x = torch.clamp(x, -1, 1)
        
        return x
    
class HeunSolver:
    """
    Implements Heun's method (improved Euler method) for solving ODEs.
    This is a second-order Runge-Kutta method that provides better
    accuracy than the basic Euler method.
    """
    def __init__(self, model, rectified_flow):
        """
        Initialize the Heun solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
        """
        self.model = model
        self.rf = rectified_flow
        
    def generate_sample(self, rna_expr, num_steps=100, device="cuda"):
        """
        Generate a sample by solving the ODE using Heun's method.
        
        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Number of steps for the Heun method
            device: The device to run the computation on
            
        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]
        
        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size
        
        # Start from random noise (t=0)
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        # Create time steps (from t=0 to t=1)
        dt = 1.0 / num_steps
        
        # Heun integration
        for i in range(num_steps):
            # Current time
            t = i * dt
            t_batch = torch.ones(batch_size, device=device) * t
            
            # Next time
            t_next = (i + 1) * dt
            t_next_batch = torch.ones(batch_size, device=device) * t_next
            
            with torch.no_grad():
                # Step 1: Euler prediction
                v1 = self.model(x, t_batch, rna_expr)
                x_pred = x + v1 * dt
                
                # Step 2: Corrector step using predicted state
                v2 = self.model(x_pred, t_next_batch, rna_expr)
                
                # Average the two velocity estimates for improved accuracy
                v_avg = 0.5 * (v1 + v2)
                
                # Update x using the average velocity
                x = x + v_avg * dt
            
            # Progress logging
            if i % (num_steps // 10) == 0:
                logger.info(f"Generation progress: {i}/{num_steps} steps (Heun method)")
        
        # Clamp to valid image range
        x = torch.clamp(x, -1, 1)
        
        return x

class DOPRI5Solver:
    """
    Implements the Dormand-Prince (DOPRI5) method for solving ODEs.
    This is a fifth-order Runge-Kutta method with adaptive step size
    control and error estimation.
    """
    def __init__(self, model, rectified_flow, rtol=1e-3, atol=1e-4, safety=0.9):
        """
        Initialize the DOPRI5 solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
            rtol: Relative tolerance for adaptive step size control
            atol: Absolute tolerance for adaptive step size control
            safety: Safety factor for step size adjustments
        """
        self.model = model
        self.rf = rectified_flow
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        
        # Butcher tableau coefficients for Dormand-Prince method
        self.a = [
            [],  # a[0] is unused
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ]
        
        self.b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]  # 5th order solution
        self.b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order solution for error estimation
        
        self.c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    def _dormand_prince_step(self, x, t, dt, rna_expr, batch_size, device):
        """
        Perform one step of the Dormand-Prince method.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step
            rna_expr: RNA expression data
            batch_size: Batch size
            device: Computation device
            
        Returns:
            Tuple of (next_state, error_estimate)
        """
        with torch.no_grad():
            # Six stages of the Dormand-Prince method
            k = []
            
            # First stage
            t_batch = torch.ones(batch_size, device=device) * t
            k.append(self.model(x, t_batch, rna_expr))
            
            # Remaining stages
            for i in range(1, 6):
                # Create intermediate state for this stage
                x_i = x.clone()
                for j in range(i):
                    x_i = x_i + dt * self.a[i][j] * k[j]
                
                # Compute new stage value
                t_i = t + self.c[i] * dt
                t_i_batch = torch.ones(batch_size, device=device) * t_i
                k.append(self.model(x_i, t_i_batch, rna_expr))
            
            # Compute final stage (7th stage)
            x_final = x.clone()
            for j in range(5):
                x_final = x_final + dt * self.a[5][j] * k[j]
            
            t_final = t + dt
            t_final_batch = torch.ones(batch_size, device=device) * t_final
            k.append(self.model(x_final, t_final_batch, rna_expr))
            
            # Compute 5th order solution
            x_next = x.clone()
            for i in range(6):
                x_next = x_next + dt * self.b[i] * k[i]
            
            # Compute 4th order solution for error estimation
            x_next_star = x.clone()
            for i in range(7):
                x_next_star = x_next_star + dt * self.b_star[i] * k[i]
            
            # Error estimate
            error = x_next - x_next_star
            
            return x_next, error
    
    def _compute_adaptive_step_size(self, error, x, dt):
        """
        Compute adaptive step size based on error estimate.
        
        Args:
            error: Error estimate
            x: Current state
            dt: Current time step
            
        Returns:
            New time step
        """
        # Scale error by tolerance
        error_ratio = torch.norm(error) / (self.atol + self.rtol * torch.norm(x))
        
        if error_ratio == 0:
            # Increase step size if error is zero
            return dt * 2.0
        
        # Standard adaptive step size formula
        dt_new = self.safety * dt * (1.0 / error_ratio) ** (1/5)
        
        # Limit step size changes
        dt_new = torch.clamp(dt_new, dt * 0.1, dt * 5.0)
        
        return dt_new
    
    def generate_sample(self, rna_expr, num_steps=100, initial_dt=0.01, device="cuda"):
        """
        Generate a sample by solving the ODE using the DOPRI5 method
        with adaptive step size control.
        
        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Maximum number of steps for the DOPRI5 method
            initial_dt: Initial time step
            device: The device to run the computation on
            
        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]
        
        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size
        
        # Start from random noise (t=0)
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)
        
        # Initialize variables for adaptive stepping
        t = 0.0
        dt = initial_dt
        step_count = 0
        
        # Create history arrays for debugging
        t_history = [t]
        dt_history = [dt]
        
        # Integration with adaptive step size
        while t < 1.0 and step_count < num_steps:
            # Ensure we don't overshoot t=1.0
            if t + dt > 1.0:
                dt = 1.0 - t
            
            # Perform one step of DOPRI5
            x_next, error = self._dormand_prince_step(x, t, dt, rna_expr, batch_size, device)
            
            # Compute new step size based on error
            dt_next = self._compute_adaptive_step_size(error, x, dt)
            
            # Accept step
            x = x_next
            t += dt
            step_count += 1
            
            # Update step size for next iteration
            dt = dt_next
            
            # Store history
            t_history.append(t)
            dt_history.append(dt)
            
            # Progress logging
            if step_count % 10 == 0:
                logger.info(f"Generation progress: t={t:.4f}, dt={dt:.6f}, steps={step_count}")
        
        logger.info(f"DOPRI5 generation completed in {step_count} steps, final t={t:.4f}")
        
        # Log adaptive step size statistics
        if len(dt_history) > 1:
            logger.info(f"Step size statistics: min={min(dt_history):.6f}, max={max(dt_history):.6f}, avg={sum(dt_history)/len(dt_history):.6f}")
        
        # Clamp to valid image range
        x = torch.clamp(x, -1, 1)
        
        return x

class SymplecticEulerSolver:
    """
    Implements the Symplectic Euler method for solving ODEs.
    This method is energy-preserving for Hamiltonian systems and
    can provide better stability for long trajectories.
    """
    def __init__(self, model, rectified_flow):
        """
        Initialize the Symplectic Euler solver.
        
        Args:
            model: The neural network model that predicts velocities
            rectified_flow: The rectified flow object
        """
        self.model = model
        self.rf = rectified_flow
        
    def generate_sample(self, rna_expr, num_steps=100, device="cuda"):
        """
        Generate a sample by solving the ODE using Symplectic Euler method.
        
        Args:
            rna_expr: RNA expression data (B, gene_dim)
            num_steps: Number of steps for the Symplectic Euler method
            device: The device to run the computation on
            
        Returns:
            Generated sample
        """
        batch_size = rna_expr.shape[0]
        
        # Get shapes from the model
        img_channels = self.model.img_channels
        img_size = self.model.img_size
        
        # For symplectic Euler, we need position and momentum variables
        # In our case, we'll treat the image as position and introduce a momentum variable
        x = torch.randn(batch_size, img_channels, img_size, img_size, device=device)  # position
        p = torch.zeros_like(x)  # momentum, initialize with zeros
        
        # Create time steps (from t=0 to t=1)
        dt = 1.0 / num_steps
        
        # Symplectic Euler integration
        for i in range(num_steps):
            # Current time
            t = i * dt
            t_batch = torch.ones(batch_size, device=device) * t
            
            with torch.no_grad():
                # Get velocity field which acts as force
                force = self.model(x, t_batch, rna_expr)
                
                # Update momentum first (half step)
                p = p + force * dt / 2
                
                # Then update position using the updated momentum
                x = x + p * dt
                
                # Finally update momentum again (completing the step)
                t_next = t + dt
                t_next_batch = torch.ones(batch_size, device=device) * t_next
                force = self.model(x, t_next_batch, rna_expr)
                p = p + force * dt / 2
            
            # Progress logging
            if i % (num_steps // 10) == 0:
                logger.info(f"Generation progress: {i}/{num_steps} steps (Symplectic Euler)")
        
        # Clamp to valid image range
        x = torch.clamp(x, -1, 1)
        
        return x