
/** @file
 *
 * @defgroup usbd_ble_uart_example main.c
 * @{
 * @ingroup  usbd_ble_uart_example
 * @brief    USBD CDC ACM over BLE application main file.
 *
 * This file contains the source code for a sample application that uses the Nordic UART service
 * and USBD CDC ACM library.
 * This application uses the @ref srvlib_conn_params module.
 *********************************************************************************************************
 *********THIS CODE IS DERIVED FROM THE USBD_BLE_UAR EXAMPLE CODE; TO EXPEDITE BLE AND USB SETUP *********
 *********************************************************************************************************
 */

// INCLUDES

#include "nrf_sdh.h"
#include "nrf_sdh_ble.h"
#include "app_timer.h"
#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
#include "ble_advdata.h"
#include "ble_advertising.h" 
#include "ble_conn_params.h"
#include "ble_nus.h"
#include "nrf_ble_gatt.h"
#include "app_usbd.h"
#include "app_usbd_cdc_acm.h"
#include "app_usbd_serial_num.h"
#include "nrfx_gpiote.h"
#include "nrfx_spi.h"
#include "nrf_gpio.h"
#include "bsp_btn_ble.h"
#include "nrf_delay.h"
#include "nrf_drv_clock.h"
#include <string.h> 
#include <stdbool.h> 
#include <stdlib.h>


#define ENDLINE_STRING "\r\n"

// value from sdk_config
#define APP_TIMER_TICKS_TO_MS(ticks) ((uint32_t)(((ticks) * 1000) / 16384))

// BLINKING INTERVALS
#define BLINK_INTERVAL_IDLE   1000  // ms
#define BLINK_INTERVAL_ADV    200   // ms
#define LED_BLINK_INTERVAL 800
APP_TIMER_DEF(m_long_press_timer);
APP_TIMER_DEF(m_adv_blink_timer);

// LED INDICATORS
#define RGB_MODE_RED    2   // GPIO for record mode
#define RGB_MODE_BLUE  3   // GPIO for livestream mode 
                          //(passive mode = purple)

#define RGB_CONN_RED    28  // GPIO for idle state
#define RGB_CONN_GREEN  29  // GPIO for USB connected
#define RGB_CONN_BLUE   30  // GPIO for BLE activity


// DEFINE BUTTONS
#define RESET_BUTTON_PIN     21      // GPIO 21 (reset button)
#define ON_BUTTON            27
#define LONG_PRESS_MS       3000     // 3s for reset
#define SHORT_PRESS_MS       300     // 300ms for BLE advertising
#define DEBOUNCE_MS          50      // Debounce delay

static volatile bool     m_button_pressed = false;
static volatile uint32_t m_button_press_time = 0;
APP_TIMER_DEF(m_button_timer);       // Timer for press duration
static volatile bool m_long_press_triggered = false;
static uint32_t m_button_press_ticks = 0;


// USB DEFINES
static void cdc_acm_user_ev_handler(app_usbd_class_inst_t const * p_inst,
                                    app_usbd_cdc_acm_user_event_t event);
static bool m_usb_host_connected = false;

#define CDC_ACM_COMM_INTERFACE  0
#define CDC_ACM_COMM_EPIN       NRF_DRV_USBD_EPIN2

#define CDC_ACM_DATA_INTERFACE  1
#define CDC_ACM_DATA_EPIN       NRF_DRV_USBD_EPIN1
#define CDC_ACM_DATA_EPOUT      NRF_DRV_USBD_EPOUT1

static char m_cdc_data_array[BLE_NUS_MAX_DATA_LEN];

APP_USBD_CDC_ACM_GLOBAL_DEF(m_app_cdc_acm,
                            cdc_acm_user_ev_handler,
                            CDC_ACM_COMM_INTERFACE,
                            CDC_ACM_DATA_INTERFACE,
                            CDC_ACM_COMM_EPIN,
                            CDC_ACM_DATA_EPIN,
                            CDC_ACM_DATA_EPOUT,
                            APP_USBD_CDC_COMM_PROTOCOL_AT_V250); // CDC_ACM class instance
#define USB_PWR_CTRL_PIN 1

// BLE DEFINES 
#define APP_BLE_CONN_CFG_TAG            1                                           /**< A tag identifying the SoftDevice BLE configuration. */

#define APP_FEATURE_NOT_SUPPORTED       BLE_GATT_STATUS_ATTERR_APP_BEGIN + 2        /**< Reply when unsupported features are requested. */

#define DEVICE_NAME                     "BCI_Device"                      /**< Name of device. Will be included in the advertising data. */
#define NUS_SERVICE_UUID_TYPE           BLE_UUID_TYPE_VENDOR_BEGIN                  /**< UUID type for the Nordic UART Service (vendor specific). */

#define APP_BLE_OBSERVER_PRIO           3                                           /**< Application's BLE observer priority. You shouldn't need to modify this value. */

#define APP_ADV_INTERVAL                64                                          /**< The advertising interval (in units of 0.625 ms. This value corresponds to 40 ms). */
#define APP_ADV_DURATION                18000                                       /**< The advertising duration (180 seconds) in units of 10 milliseconds. */


#define MIN_CONN_INTERVAL               MSEC_TO_UNITS(20, UNIT_1_25_MS)             /**< Minimum acceptable connection interval (20 ms). Connection interval uses 1.25 ms units. */
#define MAX_CONN_INTERVAL               MSEC_TO_UNITS(75, UNIT_1_25_MS)             /**< Maximum acceptable connection interval (75 ms). Connection interval uses 1.25 ms units. */
#define SLAVE_LATENCY                   0                                           /**< Slave latency. */
#define CONN_SUP_TIMEOUT                MSEC_TO_UNITS(4000, UNIT_10_MS)             /**< Connection supervisory timeout (4 seconds). Supervision Timeout uses 10 ms units. */
#define FIRST_CONN_PARAMS_UPDATE_DELAY  APP_TIMER_TICKS(5000)                       /**< Time from initiating an event (connect or start of notification) to the first time sd_ble_gap_conn_param_update is called (5 seconds). */
#define NEXT_CONN_PARAMS_UPDATE_DELAY   APP_TIMER_TICKS(30000)                      /**< Time between each call to sd_ble_gap_conn_param_update after the first call (30 seconds). */
#define MAX_CONN_PARAMS_UPDATE_COUNT    3                                           /**< Number of attempts before giving up the connection parameter negotiation. */

#define DEAD_BEEF                       0xDEADBEEF                                  /**< Value used as error code on stack dump. Can be used to identify stack location on stack unwind. */

#define UART_TX_BUF_SIZE                256                                         /**< UART TX buffer size. */
#define UART_RX_BUF_SIZE                256                                         /**< UART RX buffer size. */


BLE_NUS_DEF(m_nus, NRF_SDH_BLE_TOTAL_LINK_COUNT);                                   /**< BLE NUS service instance. */
NRF_BLE_GATT_DEF(m_gatt);                                                           /**< GATT module instance. */
BLE_ADVERTISING_DEF(m_advertising);                                                 /**< Advertising module instance. */

static bool m_is_advertising = false;

static uint16_t   m_conn_handle          = BLE_CONN_HANDLE_INVALID;                 /**< Handle of the current connection. */
static uint16_t   m_ble_nus_max_data_len = BLE_GATT_ATT_MTU_DEFAULT - 3;            /**< Maximum length of data (in bytes) that can be transmitted to the peer by the Nordic UART service module. */
static ble_uuid_t m_adv_uuids[]          =                                          /**< Universally unique service identifier. */
{
    {BLE_UUID_NUS_SERVICE, NUS_SERVICE_UUID_TYPE}
};
static char m_nus_data_array[BLE_NUS_MAX_DATA_LEN];


// SPI & ADS1299 DEFINE
#define SPI_INSTANCE  0
static const nrfx_spi_t spi = NRFX_SPI_INSTANCE(SPI_INSTANCE);

#define ADS_SCK_PIN   13
#define ADS_MISO_PIN   14
#define ADS_MOSI_PIN   15
#define ADS_CS_PIN   16
#define ADS_DRDY_PIN 17

static volatile bool data_ready = false;

// ADS1299 Register Definitions 
#define CONFIG1    0x01
#define CONFIG2    0x02
#define CONFIG3    0x03
#define CH1SET     0x04
#define CH2SET     0x05
#define CH3SET     0x06
#define CH4SET     0x07 

#define BUFFER_SIZE 100 

const uint8_t START_BYTE[] = {0xAA};
const uint8_t END_BYTE[] = {0x55};

// timestamping
typedef struct {
    uint32_t sample_id;     // 4 bytes
    uint32_t raw_timestamp; // 4 bytes, app_timer_cnt_get() returns uint32_t
    int32_t channels[4];    // 4 channels * 4 bytes/channel = 16 bytes
    uint8_t event_flags;    // 1 byte: For event markers
} __attribute__((packed)) eeg_sample_t;

static bool m_event_states[8] = {false};

// Total size of eeg_sample_t: 4 + 4 + 16 + 1 = 25 bytes.
#define FIRMWARE_PACKET_SIZE (1 + sizeof(eeg_sample_t) + 1) // 1 (START) + 25 (eeg_sample_t) + 1 (END) = 27 bytes

// MODES Switching 
typedef enum {
    MODE_PASSIVE,
    MODE_LIVESTREAM,
    MODE_RECORD
} bci_mode_t;

static bci_mode_t m_current_mode = MODE_PASSIVE;
static uint32_t g_sample_id_counter = 0; // Global counter for sample_id
static uint8_t g_firmware_event_flags = 0; // Global variable for event flags

//input mode
#define CMD_SET_INPUT_MODE   'M'  // Input mode command prefix
#define INPUT_MODE_MONOPOLAR 0
#define INPUT_MODE_DIFF      1

// Add register definitions
#define CHnSET_BIT_PD        (1 << 7)  // Power-down bit
#define CHnSET_BIT_GAIN      0x07      // Gain bits mask
#define CHnSET_BIT_MUX       0x70      // MUX bits mask (bits 6:4)



/////////////////////////////////////////////// FUNCTIONS ///////////////////////////////////////////////

void assert_nrf_callback(uint16_t line_num, const uint8_t * p_file_name){
    app_error_handler(DEAD_BEEF, line_num, p_file_name);
}

// RGB SET
static void rgb_set_mode(bci_mode_t mode){
    nrf_gpio_pin_clear(RGB_MODE_RED);
    nrf_gpio_pin_clear(RGB_MODE_BLUE);

    switch (mode)
    {
        case MODE_PASSIVE:
            nrf_gpio_pin_set(RGB_MODE_BLUE);
            break;
        case MODE_LIVESTREAM:
            nrf_gpio_pin_set(RGB_MODE_BLUE);
            nrf_gpio_pin_set(RGB_MODE_RED);
            break;
        case MODE_RECORD:
            nrf_gpio_pin_set(RGB_MODE_RED);
            break;
        default:
            break;
    }
}

static void rgb_set_connection(bool usb_connected, bool ble_connected, bool advertising){
    // Clear all connection LEDs
    nrf_gpio_pin_clear(RGB_CONN_RED);
    nrf_gpio_pin_clear(RGB_CONN_GREEN);
    nrf_gpio_pin_clear(RGB_CONN_BLUE);

    if (usb_connected)
    {
        nrf_gpio_pin_set(RGB_CONN_GREEN); // USB priority
    }
    else if (ble_connected)
    {
        nrf_gpio_pin_set(RGB_CONN_BLUE); // Solid blue
    }
    else
    {
        nrf_gpio_pin_set(RGB_CONN_RED); // Idle: not connected
    }
}

// ADS1299
void ads1299_write_reg(uint8_t reg, uint8_t val) {
    nrf_gpio_pin_clear(ADS_CS_PIN);
    
    uint8_t tx_buf[2] = {0x40 | reg, val}; // Write command format
    uint8_t rx_buf[2] = {0};
    
    nrfx_spi_xfer_desc_t xfer = NRFX_SPI_XFER_TRX(tx_buf, 2, rx_buf, 2);
    APP_ERROR_CHECK(nrfx_spi_xfer(&spi, &xfer, 0));
    
    nrf_gpio_pin_set(ADS_CS_PIN);
}

uint8_t ads1299_read_reg(uint8_t reg) {
    nrf_gpio_pin_clear(ADS_CS_PIN);
    
    uint8_t tx_buf[2] = {0x20 | reg, 0x00};
    uint8_t rx_buf[2] = {0};
    
    nrfx_spi_xfer_desc_t xfer = NRFX_SPI_XFER_TRX(tx_buf, 2, rx_buf, 2);
    APP_ERROR_CHECK(nrfx_spi_xfer(&spi, &xfer, 0));
    
    nrf_gpio_pin_set(ADS_CS_PIN);
    return rx_buf[1];
}

void set_sample_rate(uint8_t rate_code) {
    uint8_t current = ads1299_read_reg(CONFIG1);
    current = (current & 0xF8) | (rate_code & 0x07);
    ads1299_write_reg(CONFIG1, current);
}

void set_channel_gain(uint8_t channel, uint8_t gain) {
    // For ADS1299-4, channels are 0-3.
    // The ADS1299 datasheet defines CH1SET (0x04) to CH4SET (0x07) for a 4-channel device.
    if(channel < 4) { // Only iterate for channels 0 to 3
        uint8_t reg = CH1SET + channel; // CH1SET is 0x04, so CH1=0x04, CH2=0x05, etc.
        uint8_t current = ads1299_read_reg(reg);
        current = (current & 0xF8) | (gain & 0x07); // Keep other bits, set PGA_GAIN (bits 2:0)
        ads1299_write_reg(reg, current);
    } else {
        NRF_LOG_WARNING("Firmware: Attempted to set gain for invalid channel %d (ADS1299-4 only has 4 channels).", channel);
    }
}

static void set_input_mode(uint8_t mode) {
    // Read current settings for all channels
    uint8_t ch_settings[4];
    for (uint8_t ch = 0; ch < 4; ch++) {
        ch_settings[ch] = ads1299_read_reg(CH1SET + ch);
    }

    // Configure MUX bits based on mode
    for (uint8_t ch = 0; ch < 4; ch++) {
        uint8_t new_setting = ch_settings[ch];
        
        // Clear MUX bits (bits 6:4)
        new_setting &= ~CHnSET_BIT_MUX;
        
        if (mode == INPUT_MODE_DIFF) {
            // Differential pairings:
            // CH1 = IN1P - IN1N
            // CH2 = IN2P - IN2N
            // CH3 = IN3P - IN3N
            // CH4 = IN4P - IN4N
            new_setting |= (0b001 << 4); // 001 = Normal differential input
        } else {
            // Monopolar:
            // CH1 = IN1P - INCOM
            // CH2 = IN2P - INCOM
            // CH3 = IN3P - INCOM
            // CH4 = IN4P - INCOM
            new_setting |= (0b000 << 4); // 000 = Input shorted (for bias)
        }
        
        ads1299_write_reg(CH1SET + ch, new_setting);
    }
    
    // Print new configuration
    NRF_LOG_INFO("Input mode set to %s", 
                 mode ? "Differential" : "Monopolar");
    for (uint8_t ch = 0; ch < 4; ch++) {
        NRF_LOG_INFO("CH%dSET: 0x%02X", 
                     ch+1, ads1299_read_reg(CH1SET + ch));
    }
}

// Conn Param Error Handler
static void conn_params_error_handler(uint32_t nrf_error){
    APP_ERROR_HANDLER(nrf_error);
}

// Advertising blinking handler
void adv_blink_handler(void * p_context) {
    nrf_gpio_pin_toggle(RGB_CONN_BLUE);
}

// Start Advertising
static void advertising_start(void){
    if (!m_is_advertising)
    {
        uint32_t err_code = ble_advertising_start(&m_advertising, BLE_ADV_MODE_FAST);
        APP_ERROR_CHECK(err_code);
        app_timer_start(m_adv_blink_timer, APP_TIMER_TICKS(500), NULL);
        m_is_advertising = true;
    }
}

// MODE COMMAND HANDLER
static void handle_api_command(const char * cmd){

    // Start Acquisition: 'S'
    if (strcmp(cmd, "S") == 0) {
        m_current_mode = MODE_LIVESTREAM; // Livestream is the default streaming mode
        NRF_LOG_INFO("Firmware: Start Acquisition (Livestream Mode)");
        rgb_set_mode(m_current_mode);
        app_usbd_cdc_acm_write(&m_app_cdc_acm, (uint8_t*)"ACK\n", 4);
        return;
    }
    // Stop Acquisition command: 'X'
    else if (strcmp(cmd, "X") == 0) {
        m_current_mode = MODE_PASSIVE;
        NRF_LOG_INFO("Firmware: Stop Acquisition (Passive Mode)");
        rgb_set_mode(m_current_mode);
        app_usbd_cdc_acm_write(&m_app_cdc_acm, (uint8_t*)"ACK\n", 4);
        return;
    }
    // Set Device Name command: 'N<name>'
    else if (strncmp(cmd, "N", 1) == 0) {
        char new_name[BLE_GAP_DEVNAME_MAX_LEN + 1];
        strncpy(new_name, cmd + 1, BLE_GAP_DEVNAME_MAX_LEN);
        new_name[BLE_GAP_DEVNAME_MAX_LEN] = '\0'; // Ensure null-termination

        if (m_conn_handle == BLE_CONN_HANDLE_INVALID) { // Only allow if not connected
            if (m_is_advertising) { // Stop advertising if currently active
                ret_code_t err_code = sd_ble_gap_adv_stop(m_advertising.adv_handle);
                APP_ERROR_CHECK(err_code);
                m_is_advertising = false;
            }
            
            ble_gap_conn_sec_mode_t sec_mode;
            BLE_GAP_CONN_SEC_MODE_SET_OPEN(&sec_mode);
            uint32_t err_code = sd_ble_gap_device_name_set(&sec_mode, (const uint8_t *)new_name, strlen(new_name));
            APP_ERROR_CHECK(err_code);
            NRF_LOG_INFO("Firmware: Device name set to: %s", new_name);
            advertising_start(); // Restart advertising with the new name
            
        } else {
            NRF_LOG_WARNING("Firmware: Cannot set device name while connected.");
        }
        return;
    }

    // Handle Set Event State command: 'E<index><state>\n' (e.g., E01\n for event 0 True)
    else if (strncmp(cmd, "E", 1) == 0 && strlen(cmd) >= 3) {
        uint8_t event_idx = cmd[1] - '0';
        uint8_t event_state = cmd[2] - '0';
    
        if(event_idx < 8) {
            m_event_states[event_idx] = (event_state == 1);
            NRF_LOG_INFO("Event %d set to %d", event_idx, event_state);
        }
    }

    // Handle Set Sample Rate command: 'R<rate>\n' (e.g., R500\n)
    else if (strncmp(cmd, "R", 1) == 0) {
        int rate_val = atoi(cmd + 1); // Convert string to integer

        // mappings defined based on ADS1299 datasheet
        /*
        000 : 16 kSPS
        001 : 8 kSPS
        010 : 4 kSPS
        011 : 2 kSPS
        100 : 1 kSPS
        101 : 500 SPS
        110 : 250 SPS
        */
        uint8_t rate_code;
        switch (rate_val) {
            case 250: rate_code = 0x06; break; 
            case 500: rate_code = 0x05; break; 
            case 1000: rate_code = 0x04; break; 
            case 2000: rate_code = 0x03; break;
            
            default:
                NRF_LOG_WARNING("Firmware: Unsupported sample rate: %d", rate_val);
           
                return;
        }
        set_sample_rate(rate_code);
        NRF_LOG_INFO("Firmware: Sample rate set to %d SPS", rate_val);

        return;
    }

    // Handle Set Gain command: 'G<gain>\n' 
    else if (strncmp(cmd, "G", 1) == 0) {
        int gain_val = atoi(cmd + 1); // Convert string to integer
        //PGA GAIN based on data sheet
        /*
        000 : 1 
        001 : 2 
        010 : 4 
        011 : 6 
        100 : 8 
        101 : 12 
        110 : 24
        */
        uint8_t gain_code;
        switch (gain_val) {
            case 1:  gain_code = 0x00; break; 
            case 2:  gain_code = 0x01; break; 
            case 4:  gain_code = 0x02; break; 
            case 6:  gain_code = 0x03; break; 
            case 8:  gain_code = 0x04; break; 
            case 12: gain_code = 0x05; break; 
            case 24: gain_code = 0x06; break; 
            default:
                NRF_LOG_WARNING("Firmware: Unsupported gain: %d", gain_val);
               
                return;
        }
        // Apply gain to all 4 channels
        for (int i = 0; i < 4; i++) {
            set_channel_gain(i, gain_code);
        }
        NRF_LOG_INFO("Firmware: Gain set to %d", gain_val);

        return;
    }
    
    else if (strncmp(cmd, "M", 1) == 0 && strlen(cmd) >= 2) {
    uint8_t mode = cmd[1] - '0';
    if (mode == INPUT_MODE_MONOPOLAR || mode == INPUT_MODE_DIFF) {
        set_input_mode(mode);
        app_usbd_cdc_acm_write(&m_app_cdc_acm, (uint8_t*)"ACK\n", 4);
    } else {
        NRF_LOG_WARNING("Invalid input mode: %c", cmd[1]);
    }
    return;
}
    else {
        NRF_LOG_WARNING("Firmware: Unknown command received: %s", cmd);
    }
}

// Nordic UART Service Data Handler
static void nus_data_handler(ble_nus_evt_t * p_evt){
   
    if (p_evt->type == BLE_NUS_EVT_RX_DATA)
    {
        NRF_LOG_DEBUG("Received data from BLE NUS. Writing data on CDC ACM.");
        NRF_LOG_HEXDUMP_DEBUG(p_evt->params.rx_data.p_data, p_evt->params.rx_data.length);
        
        // Null-terminate the received data to treat it as a C string
        char msg_buffer[BLE_NUS_MAX_DATA_LEN + 1]; // +1 for null terminator
        uint16_t len = p_evt->params.rx_data.length;
        if (len > BLE_NUS_MAX_DATA_LEN) len = BLE_NUS_MAX_DATA_LEN; // Prevent buffer overflow
        memcpy(msg_buffer, p_evt->params.rx_data.p_data, len);
        msg_buffer[len] = '\0'; 

        // Process the command
        handle_api_command(msg_buffer);

        // Echo back to CDC ACM for debugging if USB is connected
        if (m_usb_host_connected) { // Use m_usb_host_connected to check if USB is active host
            ret_code_t ret = app_usbd_cdc_acm_write(&m_app_cdc_acm,
                                                    (uint8_t*)msg_buffer,
                                                    len);
            if(ret != NRF_SUCCESS) {
                NRF_LOG_INFO("CDC ACM unavailable for echo, data received: %s", msg_buffer);
            }
        }
    }
}

// BLE Evt Handler
static void ble_evt_handler(ble_evt_t const * p_ble_evt, void * p_context){
    uint32_t err_code;

    switch (p_ble_evt->header.evt_id)
    {
        case BLE_GAP_EVT_CONNECTED:
            app_timer_stop(m_adv_blink_timer); // clear blink
            rgb_set_connection(false, true, false);
            break;

        case BLE_GAP_EVT_DISCONNECTED:
            app_timer_stop(m_adv_blink_timer); // Add this
            rgb_set_connection(false, false, true);
            break;


        case BLE_GAP_EVT_PHY_UPDATE_REQUEST:
        {
            NRF_LOG_DEBUG("PHY update request.");
            ble_gap_phys_t const phys =
            {
                .rx_phys = BLE_GAP_PHY_AUTO,
                .tx_phys = BLE_GAP_PHY_AUTO,
            };
            err_code = sd_ble_gap_phy_update(p_ble_evt->evt.gap_evt.conn_handle, &phys);
            APP_ERROR_CHECK(err_code);
        } break;

        case BLE_GAP_EVT_SEC_PARAMS_REQUEST:
            // Pairing not supported.
            err_code = sd_ble_gap_sec_params_reply(m_conn_handle, BLE_GAP_SEC_STATUS_PAIRING_NOT_SUPP, NULL, NULL);
            APP_ERROR_CHECK(err_code);
            break;

        case BLE_GAP_EVT_DATA_LENGTH_UPDATE_REQUEST:
        {
            ble_gap_data_length_params_t dl_params;

            // Clearing the struct will effectively set members to @ref BLE_GAP_DATA_LENGTH_AUTO.
            memset(&dl_params, 0, sizeof(ble_gap_data_length_params_t));
            err_code = sd_ble_gap_data_length_update(p_ble_evt->evt.gap_evt.conn_handle, &dl_params, NULL);
            APP_ERROR_CHECK(err_code);
        } break;

        case BLE_GATTS_EVT_SYS_ATTR_MISSING:
            // No system attributes have been stored.
            err_code = sd_ble_gatts_sys_attr_set(m_conn_handle, NULL, 0, 0);
            APP_ERROR_CHECK(err_code);
            break;

        case BLE_GATTC_EVT_TIMEOUT:
            // Disconnect on GATT Client timeout event.
            err_code = sd_ble_gap_disconnect(p_ble_evt->evt.gattc_evt.conn_handle,
                                             BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
            APP_ERROR_CHECK(err_code);
            break;

        case BLE_GATTS_EVT_TIMEOUT:
            // Disconnect on GATT Server timeout event.
            err_code = sd_ble_gap_disconnect(p_ble_evt->evt.gatts_evt.conn_handle,
                                             BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
            APP_ERROR_CHECK(err_code);
            break;

        case BLE_EVT_USER_MEM_REQUEST:
            err_code = sd_ble_user_mem_reply(p_ble_evt->evt.gattc_evt.conn_handle, NULL);
            APP_ERROR_CHECK(err_code);
            break;

        case BLE_GATTS_EVT_RW_AUTHORIZE_REQUEST:
        {
            ble_gatts_evt_rw_authorize_request_t  req;
            ble_gatts_rw_authorize_reply_params_t auth_reply;

            req = p_ble_evt->evt.gatts_evt.params.authorize_request;

            if (req.type != BLE_GATTS_AUTHORIZE_TYPE_INVALID)
            {
                if ((req.request.write.op == BLE_GATTS_OP_PREP_WRITE_REQ)     ||
                    (req.request.write.op == BLE_GATTS_OP_EXEC_WRITE_REQ_NOW) ||
                    (req.request.write.op == BLE_GATTS_OP_EXEC_WRITE_REQ_CANCEL))
                {
                    if (req.type == BLE_GATTS_AUTHORIZE_TYPE_WRITE)
                    {
                        auth_reply.type = BLE_GATTS_AUTHORIZE_TYPE_WRITE;
                    }
                    else
                    {
                        auth_reply.type = BLE_GATTS_AUTHORIZE_TYPE_READ;
                    }
                    auth_reply.params.write.gatt_status = APP_FEATURE_NOT_SUPPORTED;
                    err_code = sd_ble_gatts_rw_authorize_reply(p_ble_evt->evt.gatts_evt.conn_handle,
                                                               &auth_reply);
                    APP_ERROR_CHECK(err_code);
                }
            }
        } break; // BLE_GATTS_EVT_RW_AUTHORIZE_REQUEST

        default:
            // No implementation needed.
            break;
    }
}

// GATT Evt Handler
void gatt_evt_handler(nrf_ble_gatt_t * p_gatt, nrf_ble_gatt_evt_t const * p_evt){
    if ((m_conn_handle == p_evt->conn_handle) && (p_evt->evt_id == NRF_BLE_GATT_EVT_ATT_MTU_UPDATED))
    {
        m_ble_nus_max_data_len = p_evt->params.att_mtu_effective - OPCODE_LENGTH - HANDLE_LENGTH;
        NRF_LOG_INFO("Data len is set to 0x%X(%d)", m_ble_nus_max_data_len, m_ble_nus_max_data_len);
    }
    NRF_LOG_DEBUG("ATT MTU exchange completed. central 0x%x peripheral 0x%x",
                  p_gatt->att_mtu_desired_central,
                  p_gatt->att_mtu_desired_periph);
}

// Enter sleep mode
static void sleep_mode_enter(void){
    uint32_t err_code = bsp_indication_set(BSP_INDICATE_IDLE);
    APP_ERROR_CHECK(err_code);

    // Prepare wakeup buttons.
    err_code = bsp_btn_ble_sleep_mode_prepare();
    APP_ERROR_CHECK(err_code);

    
    err_code = sd_power_system_off();
    APP_ERROR_CHECK(err_code);
}

// When Advertising
static void on_adv_evt(ble_adv_evt_t ble_adv_evt){
    switch (ble_adv_evt)
    {
        case BLE_ADV_EVT_FAST:
            app_timer_start(m_adv_blink_timer, APP_TIMER_TICKS(500), NULL);
            m_is_advertising = true;
            break;

        case BLE_ADV_EVT_IDLE:
            app_timer_stop(m_adv_blink_timer);
            m_is_advertising = false;
            break;

        default:
            break;
    }
}

// Power Manangement
static void power_manage(void){
    uint32_t err_code = sd_app_evt_wait();
    APP_ERROR_CHECK(err_code);
}

void enable_charging(bool enable) { //auto cut charging when full
    nrf_gpio_pin_write(USB_PWR_CTRL_PIN, enable); 
}

// Idle State
static void idle_state_handle(void){
    UNUSED_RETURN_VALUE(NRF_LOG_PROCESS());
    power_manage();
}

// Button handlers
static void long_press_timeout_handler(void * p_context){
    m_long_press_triggered = true;
    NRF_LOG_INFO("Long press detected, entering power off...");
    sleep_mode_enter();  
}

static void button_event_handler(uint8_t pin_no, uint8_t button_action){
    if (pin_no == ON_BUTTON)
    {
        if (button_action == APP_BUTTON_PUSH)
        {
            m_long_press_triggered = false;
            app_timer_start(m_long_press_timer, APP_TIMER_TICKS(LONG_PRESS_MS), NULL);
        }
        else if (button_action == APP_BUTTON_RELEASE)
        {
            app_timer_stop(m_long_press_timer);

            if (!m_long_press_triggered)
            {
                // Short press → trigger BLE pairing
                NRF_LOG_INFO("Short press: BLE pairing trigger");
                if (m_conn_handle == BLE_CONN_HANDLE_INVALID)
                  {
                      advertising_start(); // restart BLE advertising
                  }

            }
        }
    }
}

static void button_timer_handler(void *p_context) {
    uint32_t elapsed_ticks = app_timer_cnt_get() - m_button_press_ticks;
    uint32_t elapsed_ms = APP_TIMER_TICKS_TO_MS(elapsed_ticks);

    if (elapsed_ms >= LONG_PRESS_MS) {
        NVIC_SystemReset();  // Long press = reset
    } else if (elapsed_ms >= SHORT_PRESS_MS) {
        advertising_start();  // Short press = BLE advertising
    }
}

void check_button(void) {
    bool current_state = (nrf_gpio_pin_read(RESET_BUTTON_PIN) == 0);

    if (current_state && !m_button_pressed) {
        m_button_pressed = true;
        m_button_press_ticks = app_timer_cnt_get();  // Record start time
        app_timer_start(m_button_timer, APP_TIMER_TICKS(LONG_PRESS_MS), NULL);
    } else if (!current_state && m_button_pressed) {
        m_button_pressed = false;
        app_timer_stop(m_button_timer);

        // Check if short press occurred
        uint32_t elapsed_ticks = app_timer_cnt_get() - m_button_press_ticks;
        uint32_t elapsed_ms = APP_TIMER_TICKS_TO_MS(elapsed_ticks);
        
        if (elapsed_ms >= SHORT_PRESS_MS && elapsed_ms < LONG_PRESS_MS) {
            advertising_start();
        }
    }
}

// USB CODE 
static void cdc_acm_user_ev_handler(app_usbd_class_inst_t const * p_inst, app_usbd_cdc_acm_user_event_t event){
    
    app_usbd_cdc_acm_t const * p_cdc_acm = app_usbd_cdc_acm_class_get(p_inst);

    switch (event)
    {
        case APP_USBD_CDC_ACM_USER_EVT_PORT_OPEN:
        {
            /*Set up the first transfer*/
            ret_code_t ret = app_usbd_cdc_acm_read(&m_app_cdc_acm,
                                                   m_cdc_data_array,
                                                   1);
            UNUSED_VARIABLE(ret);
            NRF_LOG_INFO("CDC ACM port opened");
            m_usb_host_connected = true; // Indicate that a host is connected
            rgb_set_connection(true, false, false); // Green LED for USB Host
            app_timer_stop(m_adv_blink_timer); // Stop BLE advertising if active
            break;
        }

        case APP_USBD_CDC_ACM_USER_EVT_PORT_CLOSE:
            NRF_LOG_INFO("CDC ACM port closed");
            m_usb_host_connected = false; // Host disconnected
            // Check BLE connection status to set appropriate LED
            if (m_conn_handle != BLE_CONN_HANDLE_INVALID) {
                rgb_set_connection(false, true, false); // BLE connected (Solid Blue)
            } else {
                rgb_set_connection(false, false, true); // No connection, go to advertising (Red or blinking blue)
                advertising_start(); // Restart advertising if not connected via BLE
            }
            break;

        case APP_USBD_CDC_ACM_USER_EVT_TX_DONE:
            break;

        case APP_USBD_CDC_ACM_USER_EVT_RX_DONE:
        {
            ret_code_t ret;
            static uint8_t index = 0;
            // Get the length of data received in this RX_DONE event
            size_t rx_len = app_usbd_cdc_acm_rx_size(p_cdc_acm);
            
            // Append received data to m_cdc_data_array
            if (index + rx_len <= BLE_NUS_MAX_DATA_LEN) {
                // The data is already in m_cdc_data_array at the current index due to previous read calls
                index += rx_len; 
            } else {
                NRF_LOG_WARNING("CDC ACM RX buffer overflow.");
                index = 0; // Reset buffer if overflow
            }

            // Check for endline or full buffer
            if ((index > 0 && (m_cdc_data_array[index - 1] == '\n' || m_cdc_data_array[index - 1] == '\r')) ||
                (index >= (BLE_NUS_MAX_DATA_LEN))) // Using BLE_NUS_MAX_DATA_LEN as a general buffer size
            {
                // Null-terminate and process the received command
                m_cdc_data_array[index] = '\0';
                handle_api_command((char *)m_cdc_data_array); // Process the command

                index = 0; // Reset index for next command
            }

            // Set up the next read operation (read 1 byte at a time until endline)
            ret = app_usbd_cdc_acm_read(&m_app_cdc_acm,
                                        &m_cdc_data_array[index],
                                        1);
            if (ret != NRF_SUCCESS && ret != NRF_ERROR_BUSY) {
                APP_ERROR_CHECK(ret);
            }
            break;
        }
        default:
            break;
    }
}

static void usbd_user_ev_handler(app_usbd_event_type_t event){
    switch (event)
    {
        case APP_USBD_EVT_DRV_SUSPEND:
            break;

        case APP_USBD_EVT_DRV_RESUME:
            break;

        case APP_USBD_EVT_STARTED:
            break;

        case APP_USBD_EVT_STOPPED:
            app_usbd_disable();
            break;

        case APP_USBD_EVT_POWER_DETECTED:
            NRF_LOG_INFO("USB power detected");
            // This means USB is connected, but not necessarily to a host (could be a wall charger)
            // Prioritize USB power over battery
            enable_charging(true); // Ensure charging is enabled
            rgb_set_connection(false, false, false); // Clear all connection LEDs
            // If it's just power, maybe a specific charging LED?
            // For now, assume it's a host connection if power is detected and we proceed to enable USBD
            if (!nrf_drv_usbd_is_enabled())
            {
                app_usbd_enable();
            }
            break;

        case APP_USBD_EVT_POWER_REMOVED:
        {
            NRF_LOG_INFO("USB power removed");
            enable_charging(false); // Stop charging
            m_usb_host_connected = false; // Assume host is gone
            app_usbd_stop(); // Stop USB stack

            // Re-evaluate connection status for LEDs
            if (m_conn_handle != BLE_CONN_HANDLE_INVALID) {
                rgb_set_connection(false, true, false); // BLE connected (Solid Blue)
            } else {
                rgb_set_connection(false, false, true); // No connection, go to advertising (Red or blinking blue)
                advertising_start(); // Restart advertising if not connected via BLE
            }
        }
            break;

        case APP_USBD_EVT_POWER_READY:
        {
              // This event signifies that USB enumeration completed successfully, meaning a host is connected.
              m_usb_host_connected = true;
              app_usbd_start(); // Start USB application layer
              NRF_LOG_INFO("USB Host connected");
              rgb_set_connection(true, false, false); // Green LED for USB Host
              app_timer_stop(m_adv_blink_timer); // Stop BLE advertising if active
            break;
        }
        default:
            break;
    }
}

// ADS1299-4

// DRDY interrupt handler
void drdy_handler(nrfx_gpiote_pin_t pin, nrf_gpiote_polarity_t action) {
    static uint32_t last_time = 0;
    uint32_t now = app_timer_cnt_get();
    
   
    if(APP_TIMER_TICKS_TO_MS(now - last_time) > 1) {
        data_ready = true;
        last_time = now;
    }
}

void init_drdy_interrupt(void) {
    nrfx_gpiote_in_config_t config = {
        .sense = NRF_GPIOTE_POLARITY_HITOLO,
        .pull = NRF_GPIO_PIN_NOPULL,
        .is_watcher = false,
        .hi_accuracy = true,
        .skip_gpio_setup = false
    };
    
    APP_ERROR_CHECK(nrfx_gpiote_in_init(ADS_DRDY_PIN, &config, drdy_handler));
    nrfx_gpiote_in_event_enable(ADS_DRDY_PIN, true);
}

bool read_ads1299_data(eeg_sample_t* sample) {
    // Buffer size 15 for 4 channels (3 status bytes + 4 channels * 3 bytes data)
    uint8_t tx_buf[15] = {0};
    uint8_t rx_buf[15] = {0};

    nrf_gpio_pin_clear(ADS_CS_PIN);
    // Read 15 bytes: 3 status bytes + 4 channels * 3 bytes/channel = 15 bytes
    nrfx_spi_xfer_desc_t xfer = NRFX_SPI_XFER_TRX(tx_buf, 15, rx_buf, 15); 
    ret_code_t ret = nrfx_spi_xfer(&spi, &xfer, 0);
    nrf_gpio_pin_set(ADS_CS_PIN);

    if(ret != NRF_SUCCESS) {
        return false;
    }

    if(sample) {
        sample->sample_id = g_sample_id_counter++; // Assign and increment global counter
        sample->raw_timestamp = app_timer_cnt_get(); // Use current timer tick

        // Parse 4 channels (3 bytes each), starting after 3 status bytes
        for(int i = 0; i < 4; i++) { // Loop for 4 channels
            // Correctly combine 3 bytes into a 24-bit value
            int32_t val = (rx_buf[3 + 3*i] << 16) | (rx_buf[3 + 3*i + 1] << 8) | rx_buf[3 + 3*i + 2];
            
            // Sign extend 24-bit value to 32-bit if it's a negative number
            if (val & 0x00800000) { // Check if the 24th bit (MSB of 24-bit value) is set
                val |= 0xFF000000;  // Extend with leading ones
            }
            sample->channels[i] = val;
        }
    }
    return true;
}

bool ads1299_write_verify(uint8_t reg, uint8_t val) {
    ads1299_write_reg(reg, val);
    return (ads1299_read_reg(reg) == val);
}

void ads1299_burst_write(uint8_t start_reg, uint8_t *values, uint8_t count) {
    nrf_gpio_pin_clear(ADS_CS_PIN);
    uint8_t tx_buf[count+1];
    tx_buf[0] = 0x40 | start_reg;
    memcpy(&tx_buf[1], values, count);
    
    nrf_gpio_pin_set(ADS_CS_PIN);
}

void handle_stream_mode(void) {

//for both livestream and record
    if(data_ready) {
        eeg_sample_t sample;
        read_ads1299_data(&sample); // populates sample_id, timestamp, channels
        
        sample.event_flags = g_firmware_event_flags; // Set event flags from global variable
        for(uint8_t i=0; i<8; i++){
            if(m_event_states[i]){  //  event state array
                sample.event_flags |= (1 << i);
            }
        }
        // Prepare the packet with START_BYTE and END_BYTE
        uint8_t tx_buffer[FIRMWARE_PACKET_SIZE]; // Use the new defined size (27 bytes)
        tx_buffer[0] = START_BYTE[0];
        memcpy(&tx_buffer[1], &sample, sizeof(eeg_sample_t)); // Copy the 25-byte struct
        tx_buffer[FIRMWARE_PACKET_SIZE - 1] = END_BYTE[0]; //  index for END_BYTE

        if(m_usb_host_connected) { // Use m_usb_host_connected
            app_usbd_cdc_acm_write(&m_app_cdc_acm, tx_buffer, sizeof(tx_buffer));
        } 
        else if(m_conn_handle != BLE_CONN_HANDLE_INVALID) {
          uint16_t length = sizeof(tx_buffer);
          ble_nus_data_send(&m_nus, tx_buffer, &length, m_conn_handle);
        }
        data_ready = false; // Reset flag after processing
    }
}

static void update_event_states(uint8_t event_flags) {
    for(uint8_t i=0; i<8; i++) {
        m_event_states[i] = (event_flags & (1 << i));
    }
}

void handle_ads1299_modes(void) {
    switch (m_current_mode) {
        case MODE_PASSIVE:    rgb_set_mode(MODE_PASSIVE); break;
        case MODE_LIVESTREAM: handle_stream_mode(); break;
        case MODE_RECORD:     handle_stream_mode(); break;
    }
}


/////////////////////////////////////////////// INITS ///////////////////////////////////////////////

// INIT LED AND BUTTONS
static void rgb_led_init(void){
    nrf_gpio_cfg_output(RGB_MODE_RED);
    nrf_gpio_cfg_output(RGB_MODE_BLUE);

    nrf_gpio_cfg_output(RGB_CONN_RED);
    nrf_gpio_cfg_output(RGB_CONN_GREEN);
    nrf_gpio_cfg_output(RGB_CONN_BLUE);

    // Turn off all LEDs initially
    nrf_gpio_pin_clear(RGB_MODE_RED);
    nrf_gpio_pin_clear(RGB_MODE_BLUE);
    nrf_gpio_pin_clear(RGB_CONN_RED);
    nrf_gpio_pin_clear(RGB_CONN_GREEN);
    nrf_gpio_pin_clear(RGB_CONN_BLUE);
}

static void button_init(void) {
    nrf_gpio_cfg_input(RESET_BUTTON_PIN, NRF_GPIO_PIN_PULLUP);

    // Create timer for press duration tracking
    ret_code_t err_code = app_timer_create(&m_button_timer, APP_TIMER_MODE_SINGLE_SHOT, button_timer_handler);
    APP_ERROR_CHECK(err_code);
}

// GAP INIT
static void gap_params_init(void){
    uint32_t                err_code;
    ble_gap_conn_params_t   gap_conn_params;
    ble_gap_conn_sec_mode_t sec_mode;

    BLE_GAP_CONN_SEC_MODE_SET_OPEN(&sec_mode);

    err_code = sd_ble_gap_device_name_set(&sec_mode,
                                          (const uint8_t *) DEVICE_NAME,
                                          strlen(DEVICE_NAME));
    APP_ERROR_CHECK(err_code);

    memset(&gap_conn_params, 0, sizeof(gap_conn_params));

    gap_conn_params.min_conn_interval = MIN_CONN_INTERVAL;
    gap_conn_params.max_conn_interval = MAX_CONN_INTERVAL;
    gap_conn_params.slave_latency     = SLAVE_LATENCY;
    gap_conn_params.conn_sup_timeout  = CONN_SUP_TIMEOUT;

    err_code = sd_ble_gap_ppcp_set(&gap_conn_params);
    APP_ERROR_CHECK(err_code);
}

// TIMERS INIT
static void timers_init(void){
    ret_code_t err_code = app_timer_init();
    APP_ERROR_CHECK(err_code);

    err_code = app_timer_create(&m_adv_blink_timer, APP_TIMER_MODE_REPEATED, adv_blink_handler);
    APP_ERROR_CHECK(err_code);
}

// SERVICE INIT
static void services_init(void){
    uint32_t       err_code;
    ble_nus_init_t nus_init;

    memset(&nus_init, 0, sizeof(nus_init));

    nus_init.data_handler = nus_data_handler;

    err_code = ble_nus_init(&m_nus, &nus_init);
    APP_ERROR_CHECK(err_code);
}

// CONN PARAMS INIT
static void conn_params_init(void){
    uint32_t               err_code;
    ble_conn_params_init_t cp_init;

    memset(&cp_init, 0, sizeof(cp_init));

    cp_init.p_conn_params                  = NULL;
    cp_init.first_conn_params_update_delay = FIRST_CONN_PARAMS_UPDATE_DELAY;
    cp_init.next_conn_params_update_delay  = NEXT_CONN_PARAMS_UPDATE_DELAY;
    cp_init.max_conn_params_update_count   = MAX_CONN_PARAMS_UPDATE_COUNT;
    cp_init.start_on_notify_cccd_handle    = BLE_GATT_HANDLE_INVALID;
    cp_init.disconnect_on_fail             = true;
    cp_init.evt_handler                    = NULL;
    cp_init.error_handler                  = conn_params_error_handler;

    err_code = ble_conn_params_init(&cp_init);
    APP_ERROR_CHECK(err_code);
}

// SoftDevice INIT
static void ble_stack_init(void){
    ret_code_t err_code;

    err_code = nrf_sdh_enable_request();
    APP_ERROR_CHECK(err_code);

    // Configure the BLE stack using the default settings.
    // Fetch the start address of the application RAM.
    uint32_t ram_start = 0;
    err_code = nrf_sdh_ble_default_cfg_set(APP_BLE_CONN_CFG_TAG, &ram_start);
    APP_ERROR_CHECK(err_code);

    // Enable BLE stack.
    err_code = nrf_sdh_ble_enable(&ram_start);
    APP_ERROR_CHECK(err_code);

    // Register a handler for BLE events.
    NRF_SDH_BLE_OBSERVER(m_ble_observer, APP_BLE_OBSERVER_PRIO, ble_evt_handler, NULL);
}

// GATT INIT
void gatt_init(void){
    ret_code_t err_code;

    err_code = nrf_ble_gatt_init(&m_gatt, gatt_evt_handler);
    APP_ERROR_CHECK(err_code);

    err_code = nrf_ble_gatt_att_mtu_periph_set(&m_gatt, 64);
    APP_ERROR_CHECK(err_code);
}

// ADVERTISING INIT
static void advertising_init(void){
    uint32_t               err_code;
    ble_advertising_init_t init;

    memset(&init, 0, sizeof(init));

    init.advdata.name_type          = BLE_ADVDATA_FULL_NAME;
    init.advdata.include_appearance = false;
    init.advdata.flags              = BLE_GAP_ADV_FLAGS_LE_ONLY_LIMITED_DISC_MODE;

    init.srdata.uuids_complete.uuid_cnt = sizeof(m_adv_uuids) / sizeof(m_adv_uuids[0]);
    init.srdata.uuids_complete.p_uuids  = m_adv_uuids;

    init.config.ble_adv_fast_enabled  = true;
    init.config.ble_adv_fast_interval = APP_ADV_INTERVAL;
    init.config.ble_adv_fast_timeout  = APP_ADV_DURATION;

    init.evt_handler = on_adv_evt;

    err_code = ble_advertising_init(&m_advertising, &init);
    APP_ERROR_CHECK(err_code);

    ble_advertising_conn_cfg_tag_set(&m_advertising, APP_BLE_CONN_CFG_TAG);
}

// LOG INIT
static void log_init(void){
    ret_code_t err_code = NRF_LOG_INIT(NULL);
    APP_ERROR_CHECK(err_code);

    NRF_LOG_DEFAULT_BACKENDS_INIT();
}

// SPI INIT
void spi_init(void) {
    nrfx_spi_config_t spi_config = {
        .sck_pin = ADS_SCK_PIN,
        .mosi_pin = ADS_MOSI_PIN,
        .miso_pin = ADS_MISO_PIN,
        .ss_pin = NRFX_SPI_PIN_NOT_USED, // control CS manually
        .irq_priority = NRFX_SPI_DEFAULT_CONFIG_IRQ_PRIORITY,
        .orc = 0xFF,
        .frequency = NRF_SPI_FREQ_4M, 
        .mode = NRF_SPI_MODE_1, // CPOL=0, CPHA=1 for ADS1299
        .bit_order = NRF_SPI_BIT_ORDER_MSB_FIRST
    };
    
    // Initialize SPI
    ret_code_t err_code = nrfx_spi_init(&spi, &spi_config, NULL, NULL);
    APP_ERROR_CHECK(err_code);
    
    // Configure CS pin
    nrf_gpio_cfg_output(ADS_CS_PIN);
    nrf_gpio_pin_set(ADS_CS_PIN); // Start with CS high
}

void ads1299_init(void) {
    // Hardware reset
    nrf_gpio_pin_clear(ADS_CS_PIN);
    nrf_delay_us(10);
    nrf_gpio_pin_set(ADS_CS_PIN);
    nrf_delay_ms(10);
    
    // Send SDATAC command
    {
        uint8_t sdatac_cmd = 0x11;
        nrf_gpio_pin_clear(ADS_CS_PIN);
        nrfx_spi_xfer_desc_t xfer = NRFX_SPI_XFER_TX(&sdatac_cmd, 1);
        ret_code_t err_code = nrfx_spi_xfer(&spi, &xfer, 0);
        nrf_gpio_pin_set(ADS_CS_PIN);
        APP_ERROR_CHECK(err_code);
    }
    
    set_input_mode(INPUT_MODE_MONOPOLAR);

    // Configure registers
    ads1299_write_reg(CONFIG1, 0x96);
    ads1299_write_reg(CONFIG2, 0xC0);
    ads1299_write_reg(CONFIG3, 0xE0);
    ads1299_write_reg(CH1SET, 0x01);
    
    // Start continuous conversion
    {
        uint8_t start_cmd = 0x08;
        nrf_gpio_pin_clear(ADS_CS_PIN);
        nrfx_spi_xfer_desc_t xfer = NRFX_SPI_XFER_TX(&start_cmd, 1);
        ret_code_t err_code = nrfx_spi_xfer(&spi, &xfer, 0);
        nrf_gpio_pin_set(ADS_CS_PIN);
        APP_ERROR_CHECK(err_code);
    }
}

/////////////////////////////////////////////// MAIN ///////////////////////////////////////////////

int main(void)
{
    ret_code_t ret;
    static const app_usbd_config_t usbd_config = {
        .ev_state_proc = usbd_user_ev_handler
    };

    // Initialize.
    log_init();

    ret_code_t err_code = nrfx_gpiote_init();
    APP_ERROR_CHECK(err_code);

    spi_init();
    timers_init();
    button_init();
    rgb_led_init();
    rgb_set_mode(MODE_PASSIVE); // Default to passive mode
    rgb_set_connection(false, false, false); // Start with red (not connected)

    static app_button_cfg_t buttons[] = {
    {ON_BUTTON, false, NRF_GPIO_PIN_PULLUP, button_event_handler}
    };

    err_code = app_button_init(buttons, ARRAY_SIZE(buttons), APP_TIMER_TICKS(DEBOUNCE_MS));
    APP_ERROR_CHECK(err_code);

    err_code = app_button_enable();
    APP_ERROR_CHECK(err_code);

    app_usbd_serial_num_generate();

    ret = nrf_drv_clock_init();
    APP_ERROR_CHECK(ret);

    NRF_LOG_INFO("USBD BLE UART example started.");

    ret = app_usbd_init(&usbd_config);
    APP_ERROR_CHECK(ret);

    app_usbd_class_inst_t const * class_cdc_acm = app_usbd_cdc_acm_class_inst_get(&m_app_cdc_acm);
    ret = app_usbd_class_append(class_cdc_acm);
    APP_ERROR_CHECK(ret);

    ble_stack_init();
    gap_params_init();
    gatt_init();
    services_init();
    advertising_init();
    conn_params_init();

    ret = app_usbd_power_events_enable();
    APP_ERROR_CHECK(ret);


    // Enter main loop.
    for (;;) {
    if (data_ready) {
        data_ready = false;
        eeg_sample_t sample;
        
        if(read_ads1299_data(&sample)) {
            if (m_current_mode == MODE_LIVESTREAM) {
                uint16_t length = sizeof(sample);
                if (m_usb_host_connected) {
                    app_usbd_cdc_acm_write(&m_app_cdc_acm, (uint8_t*)&sample, length);
                } else if (m_conn_handle != BLE_CONN_HANDLE_INVALID) {
                    ble_nus_data_send(&m_nus, (uint8_t*)&sample, &length, m_conn_handle);
                }
            }
        }
    }
    
    check_button();
    idle_state_handle();
}
}
