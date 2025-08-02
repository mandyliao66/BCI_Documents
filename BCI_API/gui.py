import asyncio
import threading
import time
import queue
import platform
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog
from bci_device import BCIManager, BLE_AVAILABLE, NUM_CHANNELS 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque 

class BCIGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BCI-on-a-PCB Control")
        
        # Make window resizable and set minimum size instead of fixed geometry
        self.minsize(800, 600)
        self.geometry("1000x800")
        
        # Setup scrollable canvas
        self._setup_scrollable_canvas()
        
        # Rest of initialization...
        self.event_count = 2
        self.event_names = [f"Button {i+1}" for i in range(self.event_count)]
        
        self.bci_manager = BCIManager(secret_key="my_super_secret_key")
        self._configure_events()
        
        # Livestream settings
        self.livestream_console_output = tk.BooleanVar(value=True)
        self.livestream_quiet_mode = tk.BooleanVar(value=False)
        self.livestream_show_plot = tk.BooleanVar(value=False)
        self.latest_livestream_data = None
        
        # Real-time plotting setup
        self.plot_data = {f'ch_{i}': deque(maxlen=1000) for i in range(NUM_CHANNELS)}
        self.plot_timestamps = deque(maxlen=1000)
        self.fig = None
        self.axes = None
        self.canvas = None
        self.animation = None
        
        self._create_widgets()
        self._update_status_periodically()
        
        # Prompt for event count at startup
        self.after(100, self._prompt_event_count)

    def _setup_scrollable_canvas(self):
        """Setup scrollable canvas with proper width handling"""
        # Create main canvas and scrollbar for scrolling
        self.main_canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        
        # Configure scrolling with improved binding
        def configure_scroll_region(event=None):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            # Update the scrollable frame width to match canvas width minus scrollbar
            canvas_width = event.width
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        self.scrollable_frame.bind("<Configure>", configure_scroll_region)
        self.main_canvas.bind('<Configure>', configure_canvas_width)
        
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Enhanced mousewheel binding for better cross-platform support
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)  # Linux
            widget.bind("<Button-5>", self._on_mousewheel)  # Linux
        
        bind_mousewheel(self.main_canvas)
        bind_mousewheel(self)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.delta:
            # Windows and MacOS
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            # Linux
            if event.num == 4:
                self.main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.main_canvas.yview_scroll(1, "units")

    def _configure_events(self):
        """Configure events in the manager based on current settings"""
        self.bci_manager.configure_events([
            {'name': name, 'color': self._get_event_color(i)}
            for i, name in enumerate(self.event_names)
        ])

    def _get_event_color(self, index):
        """Get a distinct color for each event button"""
        colors = [
            "#FF4C4C", "#4CAF50", "#2196F3", "#FF9800", 
            "#9C27B0", "#607D8B", "#795548", "#00BCD4"
        ]
        return colors[index % len(colors)]

    def _prompt_event_count(self):
        """Show dialog to set number of events (1-8)"""
        count = simpledialog.askinteger(
            "Event Configuration",
            "Enter number of events (1-8):",
            parent=self,
            minvalue=1,
            maxvalue=8,
            initialvalue=2
        )
        
        if count is not None and 1 <= count <= 8:
            self.event_count = count
            self.event_names = [f"Button {i+1}" for i in range(self.event_count)]
        else:
            # Use defaults if user cancels
            print("DEBUG: Using default event count")
        
        # Always configure events and rebuild buttons regardless of dialog result
        self._configure_events()
        self._rebuild_event_buttons()

    def _rebuild_event_buttons(self):
        """Recreate event buttons frame with new count"""
        print(f"DEBUG: Rebuilding event buttons for {self.event_count} events")
        
        # Destroy old frame if it exists and is not None
        if hasattr(self, 'event_frame') and self.event_frame is not None:
            self.event_frame.destroy()
        
        # Create new event frame - pack it BEFORE the log frame
        self.event_frame = ttk.LabelFrame(self, text=f"Event Control ({self.event_count} buttons)", padding="10")
        
        # Pack before the log frame by packing before it
        self.event_frame.pack(padx=10, pady=5, fill="x", before=self.log_frame)
        
        self.event_buttons = []
        for i in range(self.event_count):
            print(f"DEBUG: Creating button {i+1}")
            btn = tk.Button(
                self.event_frame,
                text=f"Button {i+1}",
                font=("Arial", 12, "bold"),
                width=10,
                height=2,
                relief=tk.RAISED,
                bd=3,
                bg=self._get_event_color(i),
                activebackground=self._get_event_color(i),
                fg="black"  # Changed from "white" to "black" for better visibility
            )
            btn.bind("<ButtonPress>", lambda e, idx=i: self._set_event_state(idx, True))
            btn.bind("<ButtonRelease>", lambda e, idx=i: self._set_event_state(idx, False))
            btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            self.event_buttons.append(btn)
        
        # Configure columns for equal spacing
        for i in range(self.event_count):
            self.event_frame.columnconfigure(i, weight=1)
        
        print(f"DEBUG: Created {len(self.event_buttons)} buttons in event frame")
        
        # Force multiple updates to ensure the frame is displayed
        self.update_idletasks()
        self.update()
        
        # Log that buttons were created
        self._log_message(f"Event buttons rebuilt: {self.event_count} buttons created")

    def _create_event_frame_placeholder(self):
        """Create placeholder for event frame - will be properly built later"""
        # Don't create the frame here, just set the variable
        # _rebuild_event_buttons will handle the actual creation and packing
        self.event_frame = None
        
    def _toggle_plot_display(self):
        """Toggle the real-time plot display"""
        if self.livestream_show_plot.get():
            self._setup_plot()
            # Pack AFTER the log frame instead of before
            self.plot_frame.pack(padx=10, pady=5, fill="both", expand=False, after=self.log_frame)
        else:
            self._hide_plot()

    def _setup_plot(self):
        """Setup matplotlib plot for real-time data visualization"""
        if self.fig is not None:
            return  # Already setup
            
        self.fig, self.axes = plt.subplots(NUM_CHANNELS, 1, figsize=(10, 6), sharex=True)
        if NUM_CHANNELS == 1:
            self.axes = [self.axes]  # Make it iterable
            
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(f'Ch{i} (μV)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-100, 100)  # Initial range, will auto-adjust
            
        self.axes[-1].set_xlabel('Time (samples)')
        self.fig.tight_layout()
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self._update_plot, interval=50, blit=False)

    def _hide_plot(self):
        """Hide and cleanup the plot"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
        self.plot_frame.pack_forget()

    def _update_plot(self, frame):
        """Update the real-time plot with latest data"""
        if not self.plot_data or not self.plot_timestamps:
            return
            
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f'Ch{i} (μV)')
            
            ch_key = f'ch_{i}'
            if ch_key in self.plot_data and len(self.plot_data[ch_key]) > 0:
                y_data = list(self.plot_data[ch_key])
                x_data = list(range(len(y_data)))
                ax.plot(x_data, y_data, 'b-', linewidth=1)
                
                # Auto-adjust y-axis
                if y_data:
                    y_min, y_max = min(y_data), max(y_data)
                    y_range = y_max - y_min
                    if y_range > 0:
                        ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.1)
                        
        self.axes[-1].set_xlabel('Time (samples)')
        
    def _get_latest_data(self):
        """API method to get latest livestream data"""
        if self.latest_livestream_data:
            # Format as CSV-like string for API output
            csv_like = self._format_sample_as_csv(self.latest_livestream_data)
            self._log_message(f"Latest API Data: {csv_like}")
        else:
            self._log_message("No livestream data available")

    def _format_sample_as_csv(self, sample):
        """Format sample data to match CSV record format"""
        # Build CSV-like string matching record mode format
        parts = [
            f"sample_id:{sample['sample_id']}", 
            f"timestamp_us:{sample['raw_timestamp']}"
        ]
        
        # Add channel data
        for device in self.bci_manager.devices.values():
            if device.is_connected:
                for i in range(NUM_CHANNELS):
                    key = f"{device.device_name}_channel_{i}"
                    val = sample.get(key, 0)
                    if isinstance(val, float):
                        parts.append(f"{device.device_name}_ch{i}_uV:{val:.2f}")
                    else:
                        parts.append(f"{device.device_name}_ch{i}_uV:{val}")
        
        # Add event states using configured names
        for i in range(len(self.event_names)):
            event_key = f'event_{i}'
            is_active = sample.get(event_key, False)
            parts.append(f"{self.event_names[i]}:{int(is_active)}")
            
        return ", ".join(parts)

    def _set_event_state(self, event_index, state):
        """Set event state through manager"""
        if not self.bci_manager.devices:
            self._log_message("Warning: No devices connected")
            return
        
        success = self.bci_manager.set_global_event_state(event_index, state)
        if success:
            event_name = self.event_names[event_index]
            self._log_message(f"Event '{event_name}' set to {state}")

    def _create_device_management_frame(self):
        """Create the device management frame with proper layout"""
        device_frame = ttk.LabelFrame(self.scrollable_frame, text="Device Management", padding="10")
        device_frame.pack(padx=10, pady=5, fill="x")

        # Device List
        self.device_list_label = ttk.Label(device_frame, text="Discovered Devices:")
        self.device_list_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.device_listbox = tk.Listbox(device_frame, height=5, width=50)
        self.device_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.device_listbox.bind('<<ListboxSelect>>', self._on_device_select)

        # Connection Buttons
        self.scan_button = ttk.Button(device_frame, text="Scan BLE Devices", command=self._scan_ble_devices)
        self.scan_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.connect_button = ttk.Button(device_frame, text="Connect Selected (BLE)", command=self._connect_selected_device)
        self.connect_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.disconnect_button = ttk.Button(device_frame, text="Disconnect All", command=self._disconnect_all_devices)
        self.disconnect_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.add_sim_device_button = ttk.Button(device_frame, text="Add Simulated Device", command=self._add_simulated_device)
        self.add_sim_device_button.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        
        self.reset_simulators_button = ttk.Button(device_frame, text="Reset Simulated Devices", command=self._reset_simulated_devices)
        self.reset_simulators_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Configuration separator
        config_separator = ttk.Separator(device_frame, orient='horizontal')
        config_separator.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        # Configuration Section - using grid layout consistently
        ttk.Label(device_frame, text="Sample Rate (Hz):").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.sample_rate_var = tk.StringVar(value="250")
        sample_rate_combo = ttk.Combobox(device_frame, textvariable=self.sample_rate_var, 
                                        values=["250", "500", "1000", "2000"], 
                                        width=15, state="readonly")
        sample_rate_combo.grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(device_frame, text="ADC Gain:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.gain_var = tk.StringVar(value="24")
        gain_combo = ttk.Combobox(device_frame, textvariable=self.gain_var,
                                values=["1", "2", "4", "6", "8", "12", "24"], 
                                width=15, state="readonly")
        gain_combo.grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(device_frame, text="Device Name Prefix:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.device_name_var = tk.StringVar(value="BCI_")
        device_name_entry = ttk.Entry(device_frame, textvariable=self.device_name_var, width=15)
        device_name_entry.grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Single Input Mode Configuration (removing duplicate)
        ttk.Label(device_frame, text="Input Mode:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        input_mode_frame = ttk.Frame(device_frame)
        input_mode_frame.grid(row=9, column=1, padx=5, pady=2, sticky="w")

        self.input_mode_var = tk.StringVar(value="Monopolar")
        ttk.Radiobutton(input_mode_frame, text="Monopolar", variable=self.input_mode_var, value="Monopolar").pack(side="left")
        ttk.Radiobutton(input_mode_frame, text="Differential", variable=self.input_mode_var, value="Differential").pack(side="left", padx=(10,0))

        # Single Configuration Button (simplified from two buttons)
        self.configure_devices_button = ttk.Button(device_frame, text="Configure All Connected Devices", command=self._configure_all_devices)
        self.configure_devices_button.grid(row=10, column=0, columnspan=2, pady=10, sticky="ew")

        # Configure column weights for proper resizing
        device_frame.columnconfigure(0, weight=1)
        device_frame.columnconfigure(1, weight=2)
        
    def _create_widgets(self):
        # Create device management frame with fixed layout
        self._create_device_management_frame()

        # --- Mode Selection Frame --- (unchanged)
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="Operation Mode", padding="10")
        mode_frame.pack(padx=10, pady=5, fill="x")

        self.mode_var = tk.StringVar(value="Passive")
        self.passive_radio = ttk.Radiobutton(mode_frame, text="Passive Mode", variable=self.mode_var, value="Passive", command=self._update_mode_buttons)
        self.passive_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.livestream_radio = ttk.Radiobutton(mode_frame, text="Livestream Mode", variable=self.mode_var, value="Livestream", command=self._update_mode_buttons)
        self.livestream_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.record_radio = ttk.Radiobutton(mode_frame, text="Record Mode", variable=self.mode_var, value="Record", command=self._update_mode_buttons)
        self.record_radio.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(mode_frame, text="Record Filename:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.record_filename_entry = ttk.Entry(mode_frame, width=40)
        self.record_filename_entry.insert(0, "logs/bci_recording.csv")
        self.record_filename_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        self.start_mode_button = ttk.Button(mode_frame, text="Start Mode", command=self._start_selected_mode)
        self.start_mode_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.stop_mode_button = ttk.Button(mode_frame, text="Stop Mode", command=self._stop_current_mode)
        self.stop_mode_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        mode_frame.columnconfigure(0, weight=1)
        mode_frame.columnconfigure(1, weight=1)
        mode_frame.columnconfigure(2, weight=1)
        
        # --- Livestream Options Frame --- (unchanged)
        livestream_frame = ttk.LabelFrame(self.scrollable_frame, text="Livestream Options", padding="10")
        livestream_frame.pack(padx=10, pady=5, fill="x")
        
        self.console_output_check = ttk.Checkbutton(
            livestream_frame, 
            text="Console Output", 
            variable=self.livestream_console_output
        )
        self.console_output_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.quiet_mode_check = ttk.Checkbutton(
            livestream_frame, 
            text="Quiet Mode (API access only)", 
            variable=self.livestream_quiet_mode
        )
        self.quiet_mode_check.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.show_plot_check = ttk.Checkbutton(
            livestream_frame, 
            text="Show Real-time Plot", 
            variable=self.livestream_show_plot,
            command=self._toggle_plot_display
        )
        self.show_plot_check.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.get_latest_button = ttk.Button(
            livestream_frame, 
            text="Get Latest Data (API)", 
            command=self._get_latest_data
        )
        self.get_latest_button.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        
        livestream_frame.columnconfigure(0, weight=1)
        livestream_frame.columnconfigure(1, weight=1)
        livestream_frame.columnconfigure(2, weight=1)
        
        # Create placeholder for event frame
        self._create_event_frame_placeholder()
        
        # --- Real-time Plot Frame (initially hidden) ---
        self.plot_frame = ttk.LabelFrame(self.scrollable_frame, text="Real-time EEG Plot", padding="10")
        
        # --- Output Log Frame ---
        self.log_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Log", padding="10")
        self.log_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(padx=5, pady=5, fill="both", expand=True)
        self.log_text.config(state='disabled')

        # Queue for thread-safe GUI updates
        self.gui_queue = queue.Queue()
        self._check_gui_queue()
        self._update_mode_buttons()

    def _log_message(self, message):
        """Thread-safe logging to the GUI"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def _update_status_periodically(self):
        # Update passive mode status if no active streaming/recording
        if not self.bci_manager.is_livestreaming and not self.bci_manager.is_recording:
            status_output = self.bci_manager.passive_mode_status()
            # Only update if there's a change to avoid constant redraws
            if not hasattr(self, '_last_status_output') or self._last_status_output != status_output:
                self._log_message(status_output)
                self._last_status_output = status_output
        self.after(5000, self._update_status_periodically) # Update every 5 seconds

    def _check_gui_queue(self):
        """Check for messages from background threads"""
        while not self.gui_queue.empty():
            try:
                message = self.gui_queue.get_nowait()
                self._log_message(message)
            except queue.Empty:
                break
        self.after(100, self._check_gui_queue) # Check queue every 100ms

    def _run_async_in_thread(self, coro):
        """Helper to run an async coroutine in a new thread."""
        def run_loop(loop, coro_to_run):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro_to_run)
            loop.close()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_loop, args=(loop, coro))
        thread.daemon = True
        thread.start()

    def _scan_ble_devices(self):
        if not BLE_AVAILABLE:
            messagebox.showerror("Error", "Bleak library not available. Cannot scan for BLE devices.")
            return
        
        self.device_listbox.delete(0, tk.END)
        self._log_message("Scanning for BLE devices...")
        
        async def scan():
            try:
                from bleak import BleakScanner
                from bci_device import BCI_SERVICE_UUID
                
                devices = await BleakScanner.discover(timeout=5)
                bci_devices = [d for d in devices if BCI_SERVICE_UUID in d.metadata.get('uuids', []) or "BCI" in (d.name or "").upper()]
                
                if not bci_devices:
                    self.gui_queue.put("No BCI devices found.")
                    return
                
                self.gui_queue.put(f"Found {len(bci_devices)} BCI devices:")
                for i, d in enumerate(bci_devices):
                    display_name = d.name if d.name else "Unknown Device"
                    self.gui_queue.put(f"  {i+1}. {display_name} ({d.address})")
                    
                    # Store device info for connection
                    self.after(10, lambda dn=display_name, addr=d.address: self._add_device_to_list(dn, addr))
                    
            except Exception as e:
                self.gui_queue.put(f"BLE scan error: {str(e)}")

        self._run_async_in_thread(scan())

    def _add_device_to_list(self, display_name, address):
        """Add device to listbox in main thread"""
        device_text = f"{display_name} ({address})"
        self.device_listbox.insert(tk.END, device_text)
        
        # Store mapping for later retrieval
        if not hasattr(self.device_listbox, 'device_map'):
            self.device_listbox.device_map = {}
        self.device_listbox.device_map[device_text] = address

    def _on_device_select(self, event):
        selected_indices = self.device_listbox.curselection()
        if selected_indices:
            index = selected_indices[0]
            selected_text = self.device_listbox.get(index)
            self._log_message(f"Selected device: {selected_text}")

    def _connect_selected_device(self):
        selected_indices = self.device_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a device to connect.")
            return
        
        selected_text = self.device_listbox.get(selected_indices[0])
        
        if not hasattr(self.device_listbox, 'device_map'):
            messagebox.showerror("Error", "No device mapping found. Please scan for devices first.")
            return
            
        address = self.device_listbox.device_map.get(selected_text)

        if not address:
            messagebox.showerror("Error", "Could not retrieve device address.")
            return

        device_id = f"BLE_Device_{address.replace(':', '')}"
        self.bci_manager.add_device(device_id, connection_type='BLE', address=address)
        
        def connect_task():
            self.gui_queue.put(f"Attempting to connect to {selected_text}...")
            if self.bci_manager.connect_all(): 
                self.gui_queue.put(f"Successfully connected to {selected_text}.")
                # Use GUI settings instead of hardcoded values
                sample_rate = int(self.sample_rate_var.get())
                gain = int(self.gain_var.get())
                device_prefix = self.device_name_var.get()
                
                # Configure with GUI settings
                device = self.bci_manager.devices.get(device_id)
                if device and device.is_connected:
                    device.set_sample_rate(sample_rate)
                    device.set_gain(gain)
                    device.set_device_name(f"{device_prefix}01")
                    differential = self.input_mode_var.get() == "Differential"
                    device.set_input_mode(differential)
                    self.gui_queue.put(f"Device configured with GUI settings: {sample_rate}Hz, Gain {gain}")
            else:
                self.gui_queue.put(f"Failed to connect to {selected_text}.")

        threading.Thread(target=connect_task, daemon=True).start()

    def _disconnect_all_devices(self):
        def disconnect_task():
            self.gui_queue.put("Disconnecting all devices...")
            self.bci_manager.stop_acquisition() # Ensure acquisition is stopped first
            self.bci_manager.disconnect_all()
            self.gui_queue.put("All devices disconnected.")
            # Update UI in main thread
            self.after(10, self._update_mode_buttons)

        threading.Thread(target=disconnect_task, daemon=True).start()

    def _add_simulated_device(self):
        num_sim_devices = len([d for d in self.bci_manager.devices.values() if d.simulate_data])
        device_id = f"SimulatedBCI_{num_sim_devices + 1:02d}"
        
        def connect_task():
            self.gui_queue.put(f"Adding simulated device {device_id}...")
            
            self.bci_manager.add_device(
                device_id,
                connection_type='USB',
                simulate_data=True
            )
            
            device = self.bci_manager.devices[device_id]
            if device.connect():
                self.gui_queue.put(f"Successfully connected to simulated device {device_id}")
                # Use GUI settings instead of hardcoded values
                sample_rate = int(self.sample_rate_var.get())
                gain = int(self.gain_var.get())
                device_prefix = self.device_name_var.get()
                
                device.set_sample_rate(sample_rate)
                device.set_gain(gain)
                device.set_device_name(f"{device_prefix.rstrip('_')}_Sim{num_sim_devices + 1:02d}")
                differential = self.input_mode_var.get() == "Differential"
                device.set_input_mode(differential)
                self.gui_queue.put(f"Simulated device configured with GUI settings")
            else:
                self.gui_queue.put(f"Failed to connect to simulated device {device_id}")
                self.bci_manager.remove_device(device_id)
        
        threading.Thread(target=connect_task, daemon=True).start()
    
    def _update_mode_buttons(self):
        current_mode = self.mode_var.get()
        is_acquiring = self.bci_manager.is_livestreaming or self.bci_manager.is_recording
        
        self.start_mode_button.config(state='normal' if not is_acquiring else 'disabled')
        self.stop_mode_button.config(state='normal' if is_acquiring else 'disabled')
        self.record_filename_entry.config(state='normal' if current_mode == 'Record' else 'disabled')

        if current_mode == "Passive":
            self.start_mode_button.config(text="Enter Passive Mode (Refresh Status)")
        elif current_mode == "Livestream":
            self.start_mode_button.config(text="Start Livestream")
        elif current_mode == "Record":
            self.start_mode_button.config(text="Start Recording")

    def _start_selected_mode(self):
        if not self.bci_manager.devices:
            messagebox.showwarning("Warning", "No devices added. Please add or connect a device first.")
            return
        if not any(d.is_connected for d in self.bci_manager.devices.values()):
            messagebox.showwarning("Warning", "No devices are connected. Please connect a device first.")
            return

        mode = self.mode_var.get()
        
        if mode == "Passive":
            self._log_message("Entering Passive Mode. Monitoring connections...")
            self._update_status_periodically() # Force an immediate status update
            self._update_mode_buttons()
            return
        
        def start_task():
            self.gui_queue.put(f"Starting {mode} Mode...")
            success = False
            if mode == "Livestream":
                success = self.bci_manager.start_livestream(self._livestream_callback)
            elif mode == "Record":
                filename = self.record_filename_entry.get()
                if not filename:
                    self.gui_queue.put("Error: Please provide a filename for recording.")
                    return
                success = self.bci_manager.start_record(filename, event_column_names=self.event_names)
            
            if success:
                self.gui_queue.put(f"{mode} Mode started successfully.")
            else:
                self.gui_queue.put(f"Failed to start {mode} Mode.")
            # Update buttons from main thread
            self.after(100, self._update_mode_buttons)

        threading.Thread(target=start_task, daemon=True).start()

    def _stop_current_mode(self):
        def stop_task():
            self.gui_queue.put("Stopping current acquisition mode...")
            self.bci_manager.stop_acquisition()
            self.gui_queue.put("Acquisition stopped.")
            # Update buttons from main thread
            self.after(100, self._update_mode_buttons)

        threading.Thread(target=stop_task, daemon=True).start()

    def _livestream_callback(self, sample):
        """Callback for processing livestream data"""
        # Store latest data for API access
        self.latest_livestream_data = sample
        
        # Update plot data if plotting is enabled
        if self.livestream_show_plot.get():
            # Find first connected device for plot data
            device_name = None
            for dev_id, dev_status in self.bci_manager.get_all_statuses().items():
                if dev_status['is_connected']:
                    device_name = dev_status['device_name']
                    break
            
            if device_name:
                # Add data to plot buffers
                for i in range(NUM_CHANNELS):
                    key = f"{device_name}_channel_{i}"
                    val = sample.get(key, 0)
                    if isinstance(val, (int, float)):
                        self.plot_data[f'ch_{i}'].append(val)
                
                self.plot_timestamps.append(sample['sample_id'])
        
        # Console output (if enabled)
        if self.livestream_console_output.get():
            # Check if quiet mode is also enabled
            if self.livestream_quiet_mode.get():
                # In quiet mode, just store data without GUI display
                return
            
            # Original console-style output
            device_name = None
            for dev_id, dev_status in self.bci_manager.get_all_statuses().items():
                if dev_status['is_connected']:
                    device_name = dev_status['device_name']
                    break
            
            if not device_name:
                return

            # Build channel display using NUM_CHANNELS
            channel_parts = []
            for i in range(NUM_CHANNELS):
                key = f"{device_name}_channel_{i}"
                val = sample.get(key, 'N/A')
                if isinstance(val, float):
                    val = f"{val:.2f}μV"
                channel_parts.append(f"Ch{i}:{val}")

            # Show active events
            event_parts = []
            for i in range(len(self.event_names)):
                event_key = f'event_{i}'
                is_active = sample.get(event_key, False)
                if is_active:
                    event_parts.append(f"{self.event_names[i]}:ON")

            event_str = f" Events:[{', '.join(event_parts)}]" if event_parts else ""

            log_str = (f"ID:{sample['sample_id']}, TS:{sample['raw_timestamp']}, " +
                    ", ".join(channel_parts) + event_str)

            self.gui_queue.put(log_str)
        
        # CSV-formatted output (always available for API)
        elif not self.livestream_quiet_mode.get():
            # Show CSV-like format when console output is disabled but not in quiet mode
            csv_like = self._format_sample_as_csv(sample)
            self.gui_queue.put(f"CSV Format: {csv_like}")

    def _reset_simulated_devices(self):
        def reset_task():
            self.gui_queue.put("Resetting all simulated devices...")
            self.bci_manager.stop_acquisition()

            sim_ids = [dev_id for dev_id, dev in self.bci_manager.devices.items() if dev.simulate_data]
            for dev_id in sim_ids:
                self.bci_manager.remove_device(dev_id) 

            self.gui_queue.put(f"Removed {len(sim_ids)} simulated devices.")
            # Update UI in main thread
            self.after(10, self._update_mode_buttons)

        threading.Thread(target=reset_task, daemon=True).start()

    def _set_input_mode(self):
        """Set input mode for all connected devices"""
        differential = self.input_mode_var.get() == "Differential"
        
        def set_mode_task():
            success_count = 0
            for device in self.bci_manager.devices.values():
                if device.is_connected:
                    if device.set_input_mode(differential):
                        success_count += 1
            
            mode_name = "Differential" if differential else "Monopolar"
            self.gui_queue.put(f"Set input mode to {mode_name} for {success_count} devices")
        
        threading.Thread(target=set_mode_task, daemon=True).start()
    
    def get_latest_livestream_data(self):
        """Public API method to get latest livestream data"""
        return self.latest_livestream_data
    
    def _configure_all_devices(self):
        """Configure all connected devices with current GUI settings"""
        def config_task():
            sample_rate = int(self.sample_rate_var.get())
            gain = int(self.gain_var.get())
            device_prefix = self.device_name_var.get()
            differential = self.input_mode_var.get() == "Differential"
            
            connected_devices = [d for d in self.bci_manager.devices.values() if d.is_connected]
            
            if not connected_devices:
                self.gui_queue.put("No connected devices to configure.")
                return
            
            self.gui_queue.put(f"Configuring {len(connected_devices)} connected devices...")
            
            success_count = 0
            for i, device in enumerate(connected_devices):
                device_success = True
                
                # Set sample rate
                if not device.set_sample_rate(sample_rate):
                    self.gui_queue.put(f"Failed to set sample rate for {device.device_id}")
                    device_success = False
                
                # Set gain
                if not device.set_gain(gain):
                    self.gui_queue.put(f"Failed to set gain for {device.device_id}")
                    device_success = False
                
                # Set device name with numbering
                new_name = f"{device_prefix}{i+1:02d}"
                if not device.set_device_name(new_name):
                    self.gui_queue.put(f"Failed to set device name for {device.device_id}")
                    device_success = False
                
                # Set input mode
                if not device.set_input_mode(differential):
                    self.gui_queue.put(f"Failed to set input mode for {device.device_id}")
                    device_success = False
                
                if device_success:
                    success_count += 1
            
            mode_name = "Differential" if differential else "Monopolar"
            self.gui_queue.put(f"Configuration complete: {success_count}/{len(connected_devices)} devices configured")
            self.gui_queue.put(f"Settings: {sample_rate}Hz, Gain {gain}, {mode_name} mode, Names: {device_prefix}XX")
        
        threading.Thread(target=config_task, daemon=True).start()

    def _configure_selected_device(self):
        """Configure only the selected device from the device list"""
        selected_indices = self.device_listbox.curselection()
        if not selected_indices:
            self.gui_queue.put("No device selected. Please select a device from the list.")
            return
        
        selected_text = self.device_listbox.get(selected_indices[0])
        
        # Find the connected device that matches the selection
        target_device = None
        for device in self.bci_manager.devices.values():
            if device.is_connected and (device.device_id in selected_text or device.address in selected_text):
                target_device = device
                break
        
        if not target_device:
            self.gui_queue.put("Selected device is not connected or not found.")
            return
    
    def config_task():
        sample_rate = int(self.sample_rate_var.get())
        gain = int(self.gain_var.get())
        device_name = self.device_name_var.get().rstrip('_') + "_01"  # Single device name
        differential = self.input_mode_var.get() == "Differential"
        
        self.gui_queue.put(f"Configuring selected device: {target_device.device_id}")
        
        success = True
        if not target_device.set_sample_rate(sample_rate):
            self.gui_queue.put(f"Failed to set sample rate")
            success = False
        
        if not target_device.set_gain(gain):
            self.gui_queue.put(f"Failed to set gain")
            success = False
        
        if not target_device.set_device_name(device_name):
            self.gui_queue.put(f"Failed to set device name")
            success = False
        
        if not target_device.set_input_mode(differential):
            self.gui_queue.put(f"Failed to set input mode")
            success = False
        
        if success:
            mode_name = "Differential" if differential else "Monopolar"
            self.gui_queue.put(f"Device configured successfully: {sample_rate}Hz, Gain {gain}, {mode_name} mode, Name: {device_name}")
        else:
            self.gui_queue.put("Some configuration settings failed. Check device connection.")
    
    threading.Thread(target=config_task, daemon=True).start()
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit? This will disconnect all devices and stop acquisition."):
            # Cleanup plot if active
            self._hide_plot()
            self.bci_manager.stop_acquisition()
            self.bci_manager.disconnect_all()
            self.destroy()



if __name__ == "__main__":
    # Ensure asyncio event loop is running for Bleak operations
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the Tkinter app
    app = BCIGuiApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()