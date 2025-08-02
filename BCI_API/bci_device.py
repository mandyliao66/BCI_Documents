import serial
import serial.tools.list_ports
import asyncio
import threading
import time
import csv
import struct
import queue
import os

# Import simulation functions
from bci_simulator import generate_simulated_channel_data, generate_simulated_timestamp

# Conditional import for Bleak (BLE library)
try:
    from bleak import BleakClient, BleakScanner
    BLE_AVAILABLE = True
except ImportError:
    print("Warning: Bleak library not found. BLE functionality will be unavailable.")
    print("Install with: pip install bleak")
    BLE_AVAILABLE = False

# --- Constants and Configuration ---
# Define the communication protocol bytes and commands
START_BYTE = b'\xAA'
END_BYTE = b'\x55'

# Assuming 8 channels for the ADS1299
NUM_CHANNELS = 4
SAMPLE_PACKET_SIZE = 1 + 4 + 4 + (NUM_CHANNELS * 4) + 1

# Commands from API to Firmware
CMD_START_ACQUISITION = b'S\n'
CMD_STOP_ACQUISITION = b'X\n'
CMD_SET_SAMPLE_RATE = b'R'  # Followed by rate value (e.g., R500\n)
CMD_SET_GAIN = b'G'     # Followed by gain value (e.g., G24\n)
CMD_SET_DEVICE_NAME = b'N'  # Followed by name string (e.g., NBCI_Unit_01\n)
CMD_SET_INPUT_MODE = b'M'  # M0=monopolar, M1=differential

# Firmware Responses
RESP_ACK = b'ACK\n'
RESP_ERR_PREFIX = b'ERR'

# BLE Service and Characteristic UUIDs 
BCI_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"  # NUS service
BCI_DATA_CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # TX (device -> host)
BCI_CONTROL_CHARACTERISTIC_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # RX (host -> device)

# --- Helper Functions ---

def raw_to_microvolts(raw_value, gain=24, vref=4.5):
    """Convert raw ADC value to microvolts"""
    # ADS1299 24-bit ADC conversion
    voltage = (raw_value * vref) / (2**23 * gain)
    return voltage * 1_000_000  # Convert to microvolts

def parse_bci_packet(raw_bytes):
    if len(raw_bytes) != SAMPLE_PACKET_SIZE:
        return None
        
    channels = []
    for i in range(NUM_CHANNELS):
        byte_offset = 9 + (i * 4)  # Each channel is 4 bytes
        channels.append(struct.unpack_from('<i', raw_bytes, byte_offset)[0])
    
    return {
        'sample_id': struct.unpack_from('<I', raw_bytes, 1)[0],
        'raw_timestamp': struct.unpack_from('<I', raw_bytes, 5)[0],
        'channels': channels,
        'event_flags': raw_bytes[9 + (NUM_CHANNELS * 4)]  # Calculate position dynamically
    }
    
def validate_bci_packet(packet):
    """Comprehensive packet validation"""
    if len(packet) != SAMPLE_PACKET_SIZE:
        return False, f"Invalid length {len(packet)}!={SAMPLE_PACKET_SIZE}"
    
    if packet[0] != START_BYTE[0] or packet[-1] != END_BYTE[0]:
        return False, "Invalid start/end bytes"
    
    try:
        channels = struct.unpack_from('<iiii', packet, 9)
        if not all(-0x800000 <= ch <= 0x7FFFFF for ch in channels):
            return False, "Channel value out of 24-bit range"
            
        event_flags = packet[25]
        if event_flags > 0xFF: 
            return False, "Invalid event flags"
            
        return True, ""
    except struct.error:
        return False, "Packet unpack error"


def get_serial_ports():
    """Lists available serial ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

# --- BCIDevice Class ---

class BCIDevice:
    """
    Represents a single BCI-on-a-PCB unit and manages its communication,
    configuration, and data acquisition.
    Can operate in simulation mode.
    """
    def __init__(self, device_id, connection_type='USB', address=None, secret_key=None, simulate_data=False):
        """
        Initializes a BCIDevice instance.

        Args:
            device_id (str): A unique identifier for this device instance (e.g., "BCI_Unit_01").
            connection_type (str): 'USB' or 'BLE'. Defaults to 'USB'.
            address (str): For USB, the serial port (e.g., 'COM3' or '/dev/ttyUSB0').
                           For BLE, the device's MAC address or name.
                           If None, attempts auto-discovery.
            secret_key (str, optional): A secret key for API authentication (recommended).
            simulate_data (bool): If True, simulate data acquisition instead of connecting to hardware.
        """
        self.device_id = device_id
        self.connection_type = connection_type.upper()
        self.address = address
        self.secret_key = secret_key
        self.simulate_data = simulate_data
        self.is_connected = False
        self.is_acquiring = False
        self.serial_port = None
        self.ble_client = None
        self.data_queue = queue.Queue() # Raw incoming data packets (or simulated raw packets)
        self.processed_data_queue = queue.Queue() # Processed samples
        self.data_thread = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.current_events = {f'event_{i}': False for i in range(8)} # Initialize 8 events
        self.device_name = device_id # Default device name, can be configured remotely
        self.sample_rate = 250 # Default sample rate for simulation
        self.adc_gain = 24 # Default gain for simulation
        self._sim_sample_id_counter = 0
        self._sim_start_time = 0

        # BLE specific attributes
        self.ble_data_characteristic = None
        self.ble_control_characteristic = None

        print(f"BCIDevice '{self.device_id}' initialized ({self.connection_type} - {self.address}, Simulate: {self.simulate_data})")

    def _auth_check(self):
        """Internal method for secret key authentication."""
        if self.secret_key is not None:
            print(f"Authentication check for {self.device_id} with key: {self.secret_key[:4]}...")
            if self.secret_key != "my_super_secret_key": # Replace with actual key logic
                print(f"Authentication failed for {self.device_id}.")
                return False
        return True

    def _send_command_usb(self, command):
        """Sends a command over USB and waits for ACK/ERR."""
        if self.simulate_data:
            print(f"Simulating USB command '{command.decode().strip()}' for {self.device_id}")
            # Simulate ACK for commands
            if command.startswith(CMD_SET_SAMPLE_RATE):
                try:
                    self.sample_rate = int(command[1:-1].decode())
                    print(f"Simulated sample rate set to {self.sample_rate} Hz.")
                except ValueError:
                    pass
            elif command.startswith(CMD_SET_GAIN):
                try:
                    self.adc_gain = int(command[1:-1].decode())
                    print(f"Simulated ADC gain set to {self.adc_gain}.")
                except ValueError:
                    pass
            elif command.startswith(CMD_SET_DEVICE_NAME):
                self.device_name = command[1:-1].decode()
                print(f"Simulated device name set to '{self.device_name}'.")
            return True # Always succeed in simulation
        
        if not self.serial_port or not self.serial_port.is_open:
            print(f"Error: USB port not open for {self.device_id}.")
            return False
        try:
            self.serial_port.write(command)
            self.serial_port.flush()
            response = self.serial_port.readline().strip()
            if response == RESP_ACK.strip():
                return True
            elif response.startswith(RESP_ERR_PREFIX.strip()):
                print(f"Device {self.device_id} responded with error: {response.decode()}")
                return False
            else:
                print(f"Device {self.device_id} received unexpected response: {response.decode()}")
                return False
        except serial.SerialException as e:
            print(f"Serial communication error for {self.device_id}: {e}")
            self.is_connected = False
            return False

    async def _send_command_ble(self, command):
        """Sends a command over BLE and waits for ACK/ERR."""
        if self.simulate_data:
            print(f"Simulating BLE command '{command.decode().strip()}' for {self.device_id}")
            # Simulate ACK for commands
            if command.startswith(CMD_SET_SAMPLE_RATE):
                try:
                    self.sample_rate = int(command[1:-1].decode())
                    print(f"Simulated sample rate set to {self.sample_rate} Hz.")
                except ValueError:
                    pass
            elif command.startswith(CMD_SET_GAIN):
                try:
                    self.adc_gain = int(command[1:-1].decode())
                    print(f"Simulated ADC gain set to {self.adc_gain}.")
                except ValueError:
                    pass
            elif command.startswith(CMD_SET_DEVICE_NAME):
                self.device_name = command[1:-1].decode()
                print(f"Simulated device name set to '{self.device_name}'.")
            return True # Always succeed in simulation

        if not self.ble_client or not self.ble_client.is_connected:
            print(f"Error: BLE client not connected for {self.device_id}.")
            return False
        try:
            await self.ble_client.write_gatt_char(self.ble_control_characteristic, command, response=True)
            print(f"Command sent to {self.device_id} via BLE: {command.decode().strip()}")
            return True
        except Exception as e:
            print(f"BLE communication error for {self.device_id}: {e}")
            self.is_connected = False
            return False

    def connect(self):
        if self.is_connected:
            return True
        
        if self.simulate_data:
            # Direct simulation connection
            if self._auth_check():
                self.is_connected = True
                self.sync_device_time()
                print(f"Successfully connected to simulated device {self.device_id}")
                return True
            return False
        elif self.connection_type == 'USB':
            if self._auth_check() and self._connect_usb():
                self.sync_device_time()
                return True
        elif self.connection_type == 'BLE' and BLE_AVAILABLE:
            if self._auth_check() and asyncio.run(self._connect_ble()):
                self.sync_device_time()
                return True
        return False

    def _connect_usb(self):
        """Internal method to connect via USB."""
        port_to_connect = self.address
        if not port_to_connect:
            print(f"Attempting auto-discovery for {self.device_id} via USB...")
            available_ports = get_serial_ports()
            if not available_ports:
                print("No serial ports found.")
                return False
            print(f"Available ports: {available_ports}. Trying the first one.")
            port_to_connect = available_ports[0]

        try:
            self.serial_port = serial.Serial(port_to_connect, baudrate=115200, timeout=1)
            self.is_connected = True
            print(f"Successfully connected to {self.device_id} via USB on {port_to_connect}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to {self.device_id} via USB on {port_to_connect}: {e}")
            self.is_connected = False
            return False

    async def _connect_ble(self):
        """Internal method to connect via BLE."""
        if not BLE_AVAILABLE:
            print("Bleak is not installed. Cannot connect via BLE.")
            return False

        device_address_or_name = self.address
        if not device_address_or_name:
            print(f"Scanning for BLE devices for {self.device_id}...")
            devices = await BleakScanner.discover()
            bci_devices = [d for d in devices if BCI_SERVICE_UUID in d.metadata.get('uuids', []) or d.name == self.device_id]
            if not bci_devices:
                print(f"No BCI devices found advertising service {BCI_SERVICE_UUID} or named {self.device_id}.")
                return False
            found_device = next((d for d in bci_devices if d.name == self.device_id), bci_devices[0])
            device_address_or_name = found_device.address
            print(f"Found BLE device: {found_device.name} ({found_device.address}). Connecting...")

        try:
            self.ble_client = BleakClient(device_address_or_name)
            await self.ble_client.connect()
            self.is_connected = True
            print(f"Successfully connected to {self.device_id} via BLE on {device_address_or_name}")

            for service in self.ble_client.services:
                for char in service.characteristics:
                    if char.uuid == BCI_DATA_CHARACTERISTIC_UUID:
                        self.ble_data_characteristic = char
                    elif char.uuid == BCI_CONTROL_CHARACTERISTIC_UUID:
                        self.ble_control_characteristic = char

            if not self.ble_data_characteristic or not self.ble_control_characteristic:
                print(f"Error: Could not find required BLE characteristics for {self.device_id}.")
                await self.ble_client.disconnect()
                self.is_connected = False
                return False

            return True
        except Exception as e:
            print(f"Failed to connect to {self.device_id} via BLE on {device_address_or_name}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnects from the BCI device."""
        if not self.is_connected:
            print(f"Device {self.device_id} is not connected.")
            return

        self.stop_acquisition()

        if self.simulate_data:
            self.is_connected = False
            print(f"Disconnected from {self.device_id} in simulation mode.")
            return

        if self.connection_type == 'USB' and self.serial_port:
            try:
                self.serial_port.close()
                print(f"Disconnected from {self.device_id} via USB.")
            except serial.SerialException as e:
                print(f"Error closing USB port for {self.device_id}: {e}")
        elif self.connection_type == 'BLE' and self.ble_client:
            if BLE_AVAILABLE:
                asyncio.run(self.ble_client.disconnect())
                print(f"Disconnected from {self.device_id} via BLE.")
        self.is_connected = False

    def set_sample_rate(self, rate):
        """Sets the ADC sample rate on the device."""
        if not self.is_connected:
            print(f"Device {self.device_id} not connected. Cannot set sample rate.")
            return False
        command = CMD_SET_SAMPLE_RATE + str(rate).encode() + b'\n'
        if self.connection_type == 'USB':
            return self._send_command_usb(command)
        elif self.connection_type == 'BLE':
            return asyncio.run(self._send_command_ble(command))

    def set_gain(self, gain):
        """Sets the ADC gain on the device."""
        if not self.is_connected:
            print(f"Device {self.device_id} not connected. Cannot set gain.")
            return False
        command = CMD_SET_GAIN + str(gain).encode() + b'\n'
        if self.connection_type == 'USB':
            return self._send_command_usb(command)
        elif self.connection_type == 'BLE':
            return asyncio.run(self._send_command_ble(command))

    def set_device_name(self, name):
        """Sets the device's name (for BLE advertising and internal ID)."""
        if not self.is_connected:
            print(f"Device {self.device_id} not connected. Cannot set device name.")
            return False
        if len(name) > 16: # Limit for BLE advertising name
            print("Warning: Device name too long, may be truncated by firmware.")
        command = CMD_SET_DEVICE_NAME + name.encode() + b'\n'
        if self.connection_type == 'USB':
            success = self._send_command_usb(command)
        elif self.connection_type == 'BLE':
            success = asyncio.run(self._send_command_ble(command))
        if success:
            self.device_name = name
            print(f"Device {self.device_id} name set to '{self.device_name}'.")
        return success

    def set_event_state(self, event_index, state):
        """Sets the state of a specific event on this device"""
        if 0 <= event_index < 8:
            event_key = f'event_{event_index}'
            self.current_events[event_key] = bool(state)
            print(f"Device {self.device_id} event_{event_index} set to {state}")

    def _read_data_usb_thread(self):
        """Thread function to continuously read raw data from USB."""
        buffer = b''
        while not self.stop_event.is_set() and self.serial_port and self.serial_port.is_open:
            try:
                data = self.serial_port.read(self.serial_port.in_waiting or 1)
                if data:
                    buffer += data
                    while len(buffer) >= SAMPLE_PACKET_SIZE:
                        start_idx = buffer.find(START_BYTE)
                        if start_idx == -1:
                            buffer = b''
                            break
                        elif start_idx > 0:
                            buffer = buffer[start_idx:]
                            continue

                        if len(buffer) >= SAMPLE_PACKET_SIZE:
                            packet = buffer[:SAMPLE_PACKET_SIZE]
                            if packet[-1:] == END_BYTE:
                                self.data_queue.put(packet)
                                buffer = buffer[SAMPLE_PACKET_SIZE:]
                            else:
                                buffer = buffer[1:]
                        else:
                            break
                else:
                    time.sleep(0.001)
            except serial.SerialException as e:
                print(f"USB read error for {self.device_id}: {e}")
                self.is_acquiring = False
                self.is_connected = False
                break
            except Exception as e:
                print(f"Unexpected error in USB read thread for {self.device_id}: {e}")
                self.is_acquiring = False
                self.is_connected = False
                break
        print(f"USB data read thread for {self.device_id} stopped.")

    def _ble_notification_handler(self, characteristic, data):
        """Callback for BLE data notifications."""
        self.data_queue.put(data)

    async def _read_data_ble_loop(self):
        """Async function to manage BLE data notifications."""
        if not self.ble_client or not self.ble_client.is_connected:
            print(f"BLE client not connected for {self.device_id}. Cannot start notification loop.")
            return

        try:
            await self.ble_client.start_notify(self.ble_data_characteristic, self._ble_notification_handler)
            print(f"Started BLE notifications for {self.device_id}.")
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
            await self.ble_client.stop_notify(self.ble_data_characteristic)
            print(f"Stopped BLE notifications for {self.device_id}.")
        except Exception as e:
            print(f"Error in BLE data read loop for {self.device_id}: {e}")
            self.is_acquiring = False
            self.is_connected = False

    def _simulate_data_thread(self):
        """Thread function to simulate data acquisition"""
        # Ensure time base is initialized
        if not hasattr(self, '_sim_start_time') or self._sim_start_time == 0:
            self._sim_start_time = time.time() * 1_000_000
        
        self._sim_sample_id_counter = 0  # Reset counter
        
        while not self.stop_event.is_set():
            try:
                delay_per_sample = 1.0 / self.sample_rate
                
                # Generate timestamp relative to initialized time base
                raw_timestamp = int((time.time() * 1_000_000) - self._sim_start_time)
                channels = generate_simulated_channel_data(
                    self._sim_sample_id_counter, 
                    NUM_CHANNELS, 
                    self.adc_gain
                )
                
                self.data_queue.put({
                    'sample_id': self._sim_sample_id_counter,
                    'raw_timestamp': raw_timestamp,
                    'channels': channels
                })
                
                self._sim_sample_id_counter += 1
                time.sleep(delay_per_sample)
            except Exception as e:
                print(f"Simulation error: {str(e)}")
                break
                
        print(f"Simulated data thread for {self.device_id} stopped.")

    def _process_data_thread(self):
        """Thread function to process raw data packets into structured samples."""
        while not self.stop_event.is_set() or not self.data_queue.empty():
            try:
                raw_or_parsed_packet = self.data_queue.get(timeout=0.1)
                
                if self.simulate_data:
                    parsed_data = raw_or_parsed_packet
                else:
                    parsed_data = parse_bci_packet(raw_or_parsed_packet)
                    if parsed_data:
                        # Convert raw ADC values to microvolts
                        parsed_data['channels'] = [
                            raw_to_microvolts(ch, gain=self.adc_gain)
                            for ch in parsed_data['channels']
                        ]
                
                if parsed_data:
                    prefixed_channels = {
                        f"{self.device_name}_channel_{i}": val
                        for i, val in enumerate(parsed_data['channels'])
                    }
                    sample = {
                        'sample_id': parsed_data['sample_id'],
                        'raw_timestamp': parsed_data['raw_timestamp'],
                        **prefixed_channels,
                        **self.current_events # Add current event states
                    }
                    self.processed_data_queue.put(sample)
                self.data_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing data for {self.device_id}: {e}")
        print(f"Data processing thread for {self.device_id} stopped.")

    def start_acquisition(self):
        """Starts data acquisition from the device (Livestream/Record mode on device)."""
        if not self.is_connected:
            print(f"Device {self.device_id} not connected. Cannot start acquisition.")
            return False
        if self.is_acquiring:
            print(f"Device {self.device_id} is already acquiring data.")
            return True

        if self.simulate_data:
            self.stop_event.clear()
            self.data_thread = threading.Thread(target=self._simulate_data_thread, daemon=True)
            self.data_thread.start()
            self.processing_thread = threading.Thread(target=self._process_data_thread, daemon=True)
            self.processing_thread.start()
            self.is_acquiring = True
            print(f"Started simulated acquisition for {self.device_id}.")
            return True

        if self.connection_type == 'USB':
            success = self._send_command_usb(CMD_START_ACQUISITION)
            if success:
                self.stop_event.clear()
                self.data_thread = threading.Thread(target=self._read_data_usb_thread, daemon=True)
                self.data_thread.start()
                self.processing_thread = threading.Thread(target=self._process_data_thread, daemon=True)
                self.processing_thread.start()
                self.is_acquiring = True
                print(f"Started acquisition for {self.device_id} via USB.")
                return True
            return False
        elif self.connection_type == 'BLE':
            success = asyncio.run(self._send_command_ble(CMD_START_ACQUISITION))
            if success:
                self.stop_event.clear()
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    new_loop = asyncio.new_event_loop()
                    self.data_thread = threading.Thread(target=lambda: new_loop.run_until_complete(self._read_data_ble_loop()), daemon=True)
                else:
                    self.data_thread = threading.Thread(target=lambda: loop.run_until_complete(self._read_data_ble_loop()), daemon=True)
                self.data_thread.start()
                self.processing_thread = threading.Thread(target=self._process_data_thread, daemon=True)
                self.processing_thread.start()
                self.is_acquiring = True
                print(f"Started acquisition for {self.device_id} via BLE.")
                return True
            return False

    def stop_acquisition(self):
        """Stops data acquisition from the device."""
        if not self.is_acquiring:
            print(f"Device {self.device_id} is not acquiring data.")
            return True

        # Signal threads to stop
        self.stop_event.set()

        # First stop adding new data
        if not self.simulate_data:
            if self.connection_type == 'USB':
                self._send_command_usb(CMD_STOP_ACQUISITION)
            elif self.connection_type == 'BLE':
                asyncio.run(self._send_command_ble(CMD_STOP_ACQUISITION))

        # Set maximum time to wait for threads to stop
        timeout = 2.0  # seconds
        start_time = time.time()

        # Clear data queues with timeout protection
        while (time.time() - start_time) < timeout:
            # Process remaining data in queues
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.task_done()
                except queue.Empty:
                    break

            while not self.processed_data_queue.empty():
                try:
                    self.processed_data_queue.get_nowait()
                    self.processed_data_queue.task_done()
                except queue.Empty:
                    break

            # Check if threads have stopped
            if ((self.data_thread is None or not self.data_thread.is_alive()) and
                (self.processing_thread is None or not self.processing_thread.is_alive())):
                break

            time.sleep(0.01)  # Small sleep to prevent busy waiting

        # Forcefully stop threads if they didn't stop gracefully
        if self.data_thread and self.data_thread.is_alive():
            print(f"Warning: Data thread for {self.device_id} did not stop gracefully")
            
        if self.processing_thread and self.processing_thread.is_alive():
            print(f"Warning: Processing thread for {self.device_id} did not stop gracefully")

        # Clear references to threads
        self.data_thread = None
        self.processing_thread = None

        self.is_acquiring = False
        print(f"Stopped acquisition for {self.device_id}.")
        return True

    def get_status(self):
        """Returns the current status of the device."""
        return {
            'device_id': self.device_id,
            'device_name': self.device_name,
            'is_connected': self.is_connected,
            'is_acquiring': self.is_acquiring,
            'connection_type': self.connection_type,
            'address': self.address,
            'data_queue_size': self.data_queue.qsize(),
            'processed_data_queue_size': self.processed_data_queue.qsize(),
            'simulate_data': self.simulate_data,
            'sample_rate': self.sample_rate,
            'adc_gain': self.adc_gain
        }
    
    def set_input_mode(self, differential=False):
        """Sets input mode (monopolar/differential)"""
        mode = 1 if differential else 0
        cmd = CMD_SET_INPUT_MODE + str(mode).encode() + b'\n'
        if self.connection_type == 'USB':
            return self._send_command_usb(cmd)
        elif self.connection_type == 'BLE':
            return asyncio.run(self._send_command_ble(cmd))
    
    def sync_events_to_device(self):
        """Sync all current events to device"""
        event_byte = 0
        for i in range(8):  # Firmware supports max 8 events
            event_byte |= (int(self.current_events.get(f'event_{i}', False)) << i)
        cmd = f"E{event_byte:08b}\n".encode()  # Send as binary string
        if self.connection_type == 'USB':
            self._send_command_usb(cmd)
        else:
            asyncio.run(self._send_command_ble(cmd))
                
    def sync_device_time(self):
        """Synchronize device's internal clock with host time"""
        if not self.is_connected:
            print(f"{self.device_id} not connected - cannot sync time")
            return False
        
        if self.simulate_data:
            # Initialize simulation time base
            self._sim_start_time = time.time() * 1_000_000
            self._sim_sample_id_counter = 0  # Reset sample counter
            print(f"Simulated time sync for {self.device_id}")
            return True
        
        timestamp_ms = int(time.time() * 1000)
        cmd = f"T{timestamp_ms}\n".encode()
        
        if self.connection_type == 'USB':
            return self._send_command_usb(cmd)
        elif self.connection_type == 'BLE':
            return asyncio.run(self._send_command_ble(cmd))

# --- BCIManager Class ---

class BCIManager:
    """
    Manages multiple BCIDevice instances, enabling virtual daisy chaining
    and synchronized operations.
    """
    def __init__(self, secret_key=None):
        """
        Initializes the BCIManager.

        Args:
            secret_key (str, optional): A secret key to apply to all managed devices.
        """
        self.devices = {} # {device_id: BCIDevice_instance}
        self.secret_key = secret_key
        self.is_livestreaming = False
        self.is_recording = False
        self.record_file = None
        self.csv_writer = None
        self.header_written = False
        self.record_stop_event = threading.Event()
        self.record_thread = None
        self.livestream_callback = None
        self.last_sample_id = -1 # To track sample IDs for consistency
        self.event_config = {}  # {'event_0': {'name': 'Button 1', 'color': '#FF0000'}, ...}
        self.active_events = {f'event_{i}': False for i in range(8)} # Initialize 8 events
        
    def add_device(self, device_id, connection_type='USB', address=None, simulate_data=False):
        """
        Adds a BCI device to be managed.

        Args:
            device_id (str): A unique identifier for this device.
            connection_type (str): 'USB' or 'BLE'.
            address (str, optional): Connection address.
            simulate_data (bool): If True, simulate data acquisition.
        """
        if device_id in self.devices:
            print(f"Device '{device_id}' already added.")
            return
        device = BCIDevice(device_id, connection_type, address, self.secret_key, simulate_data)
        self.devices[device_id] = device
        print(f"Device '{device_id}' added to manager.")

    def remove_device(self, device_id):
        """Removes a BCI device from management."""
        if device_id not in self.devices:
            print(f"Device '{device_id}' not found.")
            return
        self.devices[device_id].disconnect()
        del self.devices[device_id]
        print(f"Device '{device_id}' removed from manager.")

    def connect_all(self):
        """Connects to all added BCI devices."""
        print("Attempting to connect to all devices...")
        success_count = 0
        for device_id, device in self.devices.items():
            if device.connect():
                success_count += 1
        print(f"Connected to {success_count}/{len(self.devices)} devices.")
        return success_count == len(self.devices)

    def disconnect_all(self):
        """Disconnects from all managed BCI devices."""
        if not any(d.is_connected for d in self.devices.values()):
            print("No devices are currently connected.")
            return

        print("Disconnecting from all devices...")
        self.stop_acquisition()
        
        for device in self.devices.values():
            if device.is_connected:
                device.disconnect()

    def configure_all(self, sample_rate=None, gain=None, device_name_prefix=None):
        """
        Configures all connected devices with specified parameters.
        Args:
            sample_rate (int, optional): Sample rate (e.g., 250, 500, 1000).
            gain (int, optional): ADC gain (e.g., 1, 24).
            device_name_prefix (str, optional): Prefix for device names (e.g., "BCI_").
                                                Each device will be named 'BCI_Unit_01', 'BCI_Unit_02', etc.
        """
        print("Configuring all connected devices...")
        for i, (device_id, device) in enumerate(self.devices.items()):
            if device.is_connected:
                if sample_rate:
                    device.set_sample_rate(sample_rate)
                if gain:
                    device.set_gain(gain)
                if device_name_prefix:
                    new_name = f"{device_name_prefix}{i+1:02d}"
                    device.set_device_name(new_name)
        print("Configuration complete.")

    def set_global_event_state(self, event_index, state):
        """
        Sets the boolean state for a custom event across all managed devices.
        This state will be appended to subsequent samples from all devices.
        Args:
            event_index (int): The index of the event (0-7).
            state (bool): True or False.
        """
        if not (0 <= event_index < 8):
            print(f"Invalid event index {event_index}. Must be 0-7.")
            return False
            
        event_key = f'event_{event_index}'
        self.active_events[event_key] = bool(state)
        
        # Sync to all devices
        for device in self.devices.values():
            device.set_event_state(event_index, state)
        
        return True

    def configure_events(self, event_definitions):
        """
        Configure event definitions.
        Args:
            event_definitions: List of dicts [
                {'name': 'Button 1', 'color': '#FF0000'},
                {'name': 'Button 2', 'color': '#00FF00'},
                ...
            ]
        """
        self.event_config.clear()
        
        for i, event_def in enumerate(event_definitions):
            if i >= 8:  # Max 8 events
                break
            event_key = f'event_{i}'
            self.event_config[event_key] = event_def

    def _acquisition_loop(self):
        """
        Main loop for livestreaming or recording data from all devices.
        This runs in a separate thread.
        """
        self.last_sample_id = -1 # Reset for new acquisition session
        
        while not self.record_stop_event.is_set() or any(not d.processed_data_queue.empty() for d in self.devices.values()):
            # Try to get one sample from each device
            current_samples = {}
            for device_id, device in self.devices.items():
                try:
                    sample = device.processed_data_queue.get(timeout=0.005) 
                    current_samples[device_id] = sample
                    device.processed_data_queue.task_done()
                except queue.Empty:
                    continue

            if not current_samples:
                time.sleep(0.001) 
                continue

            # Find the sample with the lowest sample_id
            primary_device_id = min(current_samples, key=lambda k: current_samples[k]['sample_id'])
            combined_sample = current_samples[primary_device_id].copy()

            # Increment sample_id for the combined output
            self.last_sample_id += 1
            combined_sample['sample_id'] = self.last_sample_id

            # Merge channel data from all other devices
            for device_id, sample in current_samples.items():
                if device_id == primary_device_id:
                    continue
                for key, value in sample.items():
                    if 'channel_' in key:
                        combined_sample[key] = value

            # Add event states from manager (which are synced to devices)
            for event_key in self.active_events:
                combined_sample[event_key] = self.active_events[event_key]

            # Ensure all expected channel columns are present, fill with NaN if missing
            expected_channel_cols = []
            for device in self.devices.values():
                for i in range(NUM_CHANNELS):
                    expected_channel_cols.append(f"{device.device_name}_channel_{i}")

            for col in expected_channel_cols:
                if col not in combined_sample:
                    combined_sample[col] = float('nan')

            # Livestream output
            if self.is_livestreaming and self.livestream_callback:
                self.livestream_callback(combined_sample)

            # Record output
            if self.is_recording and self.csv_writer:
                if not self.header_written:
                    header = ['sample_id', 'timestamp_us']
                    # Ensure consistent channel order
                    for dev in sorted(self.devices.values(), key=lambda d: d.device_name):
                        header += [f"{dev.device_name}_ch{i}_uV" for i in range(NUM_CHANNELS)]
                    # Add event columns using configured names
                    header += [self.event_config[f'event_{i}']['name'] for i in range(len(self.event_config))]
                    
                    self.csv_writer.fieldnames = header
                    self.csv_writer.writeheader()
                    self.header_written = True
                
                # Build row with microvolts
                row = {
                    'sample_id': combined_sample['sample_id'],
                    'timestamp_us': combined_sample['raw_timestamp']
                }
                
                # Add channel data (already converted to microvolts in simulation)
                for dev in self.devices.values():
                    for i in range(NUM_CHANNELS):
                        key = f"{dev.device_name}_channel_{i}"
                        row[f"{dev.device_name}_ch{i}_uV"] = combined_sample.get(key, 0)
                
                # Add event states using configured names
                for i, (event_key, event_def) in enumerate(self.event_config.items()):
                    row[event_def['name']] = int(self.active_events.get(event_key, False))
                
                self.csv_writer.writerow(row)

        print("Acquisition loop stopped.")

    def start_livestream(self, callback_func):
        if not self.devices:
            print("No devices added to manager.")
            return False
        if self.is_livestreaming or self.is_recording:
            print("Already livestreaming or recording. Stop current operation first.")
            return False

        self.livestream_callback = callback_func
        self.is_livestreaming = True
        self.record_stop_event.clear()

        for device in self.devices.values():
            if not device.start_acquisition():
                print(f"Failed to start acquisition for {device.device_id}. Stopping all.")
                self.stop_acquisition()
                return False

        self.record_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.record_thread.start()
        return True

    def start_record(self, filename, event_column_names=None):
        """
        Starts recording data from all connected devices to a CSV file.
        Args:
            filename (str): Path to the output CSV file.
            event_column_names (list, optional): List of string names for boolean event columns
                                                (e.g., ['Button 1', 'Button 2']).
        """
        if not self.devices:
            print("No devices added to manager.")
            return False
        if self.is_livestreaming or self.is_recording:
            print("Already livestreaming or recording. Stop current operation first.")
            return False

        # Configure events based on provided names
        if event_column_names:
            event_defs = [{'name': name, 'color': self._get_default_color(i)} 
                         for i, name in enumerate(event_column_names)]
            self.configure_events(event_defs)

        try:
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            self.record_file = open(filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.record_file, fieldnames=[]) 
            self.header_written = False
            print(f"Recording to: {filename}")
        except IOError as e:
            print(f"Error opening file for recording: {e}")
            return False

        self.is_recording = True
        self.record_stop_event.clear()
            
        for device in self.devices.values():
            if not device.start_acquisition():
                print(f"Failed to start acquisition for {device.device_id}. Stopping all.")
                self.stop_acquisition()
                return False

        self.record_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.record_thread.start()
        print("Recording started.")
        return True

    def _get_default_color(self, index):
        """Get default color for event buttons"""
        colors = [
            "#FF4C4C", "#4CAF50", "#2196F3", "#FF9800", 
            "#9C27B0", "#607D8B", "#795548", "#00BCD4"
        ]
        return colors[index % len(colors)]

    def stop_acquisition(self):
        """Stops any active livestreaming or recording and device acquisition."""
        if not self.is_livestreaming and not self.is_recording and not any(d.is_acquiring for d in self.devices.values()):
            print("No acquisition is currently active.")
            return True

        self.record_stop_event.set()
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=5)

        for device in self.devices.values():
            device.stop_acquisition()

        if self.record_file:
            self.record_file.close()
            self.record_file = None
            self.csv_writer = None
            print("Recording file closed.")

        self.is_livestreaming = False
        self.is_recording = False
        self.livestream_callback = None
        print("Acquisition stopped.")
        return True

    def get_all_statuses(self):
        """Returns a dictionary of statuses for all managed devices."""
        return {device_id: device.get_status() for device_id, device in self.devices.items()}

    def passive_mode_status(self):
        """
        Provides real-time indication of active connections in passive mode.
        """
        statuses = self.get_all_statuses()
        status_str = "\n--- Passive Mode Status ---\n"
        if not statuses:
            status_str += "No devices managed.\n"
            return status_str
        for dev_id, status in statuses.items():
            connection_status = "Connected" if status['is_connected'] else "Disconnected"
            acquisition_status = "Acquiring" if status['is_acquiring'] else "Idle"
            status_str += f"  Device '{status['device_name']}' ({dev_id}): {connection_status}, Mode: {acquisition_status}\n"
        status_str += "---------------------------\n"
        return status_str