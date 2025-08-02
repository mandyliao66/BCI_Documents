import numpy as np
import struct
import time

# --- Constants for Packet Structure ---
START_BYTE = b'\xAA'
END_BYTE = b'\x55'
NUM_CHANNELS = 4 
SAMPLE_PACKET_SIZE = 1 + 4 + 4 + (NUM_CHANNELS * 4) + 1 # Total packet size

def generate_simulated_channel_data(sample_id, num_channels, adc_gain):
    """
    Generates more realistic simulated neural data with:
    - Different frequency bands (alpha, beta, gamma)
    - Random spikes (1% chance)
    - Baseline drift
    - Returns data already in microvolts for simulation mode
    """
    channels = []
    base_freq = 0.1  # Base oscillation frequency
    
    for i in range(num_channels):
        # Base oscillation (alpha band ~8-12Hz simulated)
        alpha = np.sin(sample_id * base_freq + i * 0.5) * 30.0  # 30μV amplitude
        
        # Higher frequency components (beta/gamma)
        beta = np.sin(sample_id * base_freq * 4) * 10.0  # 10μV amplitude
        gamma = np.sin(sample_id * base_freq * 8) * 5.0   # 5μV amplitude
        
        # Random spikes (1% chance)
        spike = 0
        if np.random.random() < 0.01:
            spike = np.random.uniform(50.0, 200.0)  # Large spike in μV
            
        # Slow baseline drift
        drift = np.sin(sample_id * 0.001) * 20.0  # 20μV drift
        
        # Combine components with noise (already in microvolts)
        sim_val = (alpha + beta + gamma + drift + spike + 
                  np.random.randn() * 3.0)  # 3μV noise
        
        channels.append(float(sim_val))
    return channels

def generate_simulated_timestamp(start_time_us, jitter_ms=2):
    """
    Generates timestamp with optional jitter to simulate real hardware variability
    """
    base_time = (time.time() * 1_000_000) - start_time_us
    if jitter_ms > 0:
        jitter_us = np.random.uniform(-jitter_ms, jitter_ms) * 1000
        base_time += jitter_us
    return int(base_time)

def pack_simulated_packet(sample_id, raw_timestamp, channels):
    """
    Packs simulated data into a byte packet, mimicking the hardware format.
    This is useful if you want to test the `parse_bci_packet` function.
    """
    packet = START_BYTE
    packet += struct.pack('<I', sample_id)
    packet += struct.pack('<I', raw_timestamp)
    
    # Pack channels as 24-bit signed integers (simulating ADC output)
    for channel_val in channels:
        # Convert microvolts back to raw ADC values for packet simulation
        raw_adc = int(channel_val * (2**23) / (4.5 * 1_000_000))  # Reverse conversion
        raw_adc = max(-0x800000, min(0x7FFFFF, raw_adc))  # Clamp to 24-bit range
        packet += struct.pack('<i', raw_adc)
    
    packet += b'\x00'  # Event flags byte
    packet += END_BYTE
    
    if len(packet) != SAMPLE_PACKET_SIZE:
        print(f"Warning: Simulated packet size mismatch! Expected {SAMPLE_PACKET_SIZE}, got {len(packet)}")
    
    return packet

class BCISimulator:
    """Enhanced BCI simulator class for more realistic data generation"""
    
    def __init__(self, sample_rate=250, adc_gain=24, num_channels=NUM_CHANNELS, noise_level=3.0):
        self.sample_rate = sample_rate
        self.adc_gain = adc_gain
        self.num_channels = num_channels
        self.noise_level = noise_level  # μV
        self.start_time = time.time() * 1_000_000
        self.sample_id = 0
        
        # Parameters for realistic EEG simulation
        self.alpha_freq = 10.0  # Hz
        self.beta_freq = 20.0   # Hz
        self.gamma_freq = 40.0  # Hz
        
        # Channel-specific parameters
        self.channel_offsets = np.random.uniform(0, 2*np.pi, num_channels)
        self.channel_gains = np.random.uniform(0.8, 1.2, num_channels)
        
    def generate_sample(self):
        """Generate a complete sample with timestamp and realistic EEG data"""
        ts = generate_simulated_timestamp(self.start_time)
        
        # Time in seconds for frequency calculations
        t = self.sample_id / self.sample_rate
        
        channels = []
        for i in range(self.num_channels):
            # Multi-band EEG simulation
            alpha = np.sin(2 * np.pi * self.alpha_freq * t + self.channel_offsets[i]) * 30.0
            beta = np.sin(2 * np.pi * self.beta_freq * t + self.channel_offsets[i]) * 15.0
            gamma = np.sin(2 * np.pi * self.gamma_freq * t + self.channel_offsets[i]) * 8.0
            
            # Add 1/f noise (pink noise approximation)
            pink_noise = np.random.randn() * self.noise_level / (1 + i * 0.1)
            
            # Occasional artifacts
            artifact = 0
            if np.random.random() < 0.005:  # 0.5% chance
                artifact = np.random.uniform(-100, 100)
            
            # Combine all components
            signal = (alpha + beta + gamma + pink_noise + artifact) * self.channel_gains[i]
            channels.append(float(signal))
        
        self.sample_id += 1
        return {
            'sample_id': self.sample_id,
            'timestamp': ts,
            'channels': channels,
            'packet': pack_simulated_packet(self.sample_id, ts, channels)
        }

    def reset(self):
        """Reset simulator state"""
        self.start_time = time.time() * 1_000_000
        self.sample_id = 0

if __name__ == "__main__":
    print("--- Testing Enhanced BCI Simulator ---")
    
    # Test the enhanced simulator
    simulator = BCISimulator(sample_rate=250, adc_gain=24)
    
    print(f"Generating {NUM_CHANNELS}-channel EEG simulation...")
    print("Channel data is in microvolts (μV)")
    
    try:
        for i in range(10):
            sample = simulator.generate_sample()
            
            print(f"\nSample {sample['sample_id']}:")
            print(f"Timestamp: {sample['timestamp']} μs")
            print(f"Channels (μV): {[f'{c:.2f}' for c in sample['channels']]}")
            print(f"Packet size: {len(sample['packet'])} bytes")
            
            # Validate packet structure
            if len(sample['packet']) == SAMPLE_PACKET_SIZE:
                print("✓ Packet structure valid")
            else:
                print("✗ Packet structure invalid")
            
            time.sleep(1.0 / simulator.sample_rate)
        
        print(f"\n--- Simulation Statistics ---")
        print(f"Sample rate: {simulator.sample_rate} Hz")
        print(f"ADC gain: {simulator.adc_gain}x")
        print(f"Channels: {simulator.num_channels}")
        print(f"Noise level: {simulator.noise_level} μV")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {str(e)}")
    
    print("--- Enhanced Simulation Test Complete ---")