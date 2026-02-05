/*
 * High-speed LSM6DSOX IMU streaming for Arduino Nano RP2040 Connect
 * 
 * Output format (binary, 14 bytes per sample):
 *   [0xAA] [0x55] [ax_L] [ax_H] [ay_L] [ay_H] [az_L] [az_H] 
 *                 [gx_L] [gx_H] [gy_L] [gy_H] [gz_L] [gz_H]
 * 
 * Sync bytes 0xAA 0x55 let you realign the stream if bytes are dropped.
 * All values are raw int16_t, little-endian (native RP2040).
 * 
 * Accel: ±4g  → 0.122 mg/LSB
 * Gyro:  ±2000°/s → 70 mdps/LSB
 */

#include <Arduino.h>
#include <Wire.h>

// On Nano RP2040 Connect with Mbed core, the onboard IMU is on Wire1
// Wire1 uses internal pins: SDA = p12 (GPIO12), SCL = p13 (GPIO13)
// We need to explicitly create the Wire1 instance
MbedI2C Wire1(p12, p13);

// LSM6DSOX I2C address (directly on nano RP2040 connect)
#define IMU_ADDR 0x6A

// Register addresses
#define REG_CTRL1_XL    0x10  // Accel control
#define REG_CTRL2_G     0x11  // Gyro control
#define REG_CTRL3_C     0x12  // Control register 3
#define REG_CTRL4_C     0x13  // Control register 4
#define REG_CTRL6_C     0x15  // Control register 6
#define REG_CTRL7_G     0x16  // Control register 7
#define REG_STATUS      0x1E  // Status register
#define REG_OUTX_L_G    0x22  // Gyro X low byte (start of 12-byte burst)
#define REG_WHO_AM_I    0x0F

// Status bits
#define STATUS_XLDA     0x01  // Accel data available
#define STATUS_GDA      0x02  // Gyro data available

// Packet sync bytes
#define SYNC_BYTE_1     0xAA
#define SYNC_BYTE_2     0x55

// Output buffer: sync(2) + gyro(6) + accel(6) = 14 bytes
uint8_t txBuffer[14];

void writeReg(uint8_t reg, uint8_t val) {
    Wire1.beginTransmission(IMU_ADDR);
    Wire1.write(reg);
    Wire1.write(val);
    Wire1.endTransmission();
}

uint8_t readReg(uint8_t reg) {
    Wire1.beginTransmission(IMU_ADDR);
    Wire1.write(reg);
    Wire1.endTransmission(false);
    Wire1.requestFrom(IMU_ADDR, (uint8_t)1);
    return Wire1.read();
}

void burstRead(uint8_t startReg, uint8_t* buf, uint8_t len) {
    Wire1.beginTransmission(IMU_ADDR);
    Wire1.write(startReg);
    Wire1.endTransmission(false);
    Wire1.requestFrom(IMU_ADDR, len);
    for (uint8_t i = 0; i < len && Wire1.available(); i++) {
        buf[i] = Wire1.read();
    }
}

void setup() {
    // USB Serial at max practical rate
    Serial.begin(2000000);
    while (!Serial && millis() < 3000); // Wait up to 3s for serial monitor
    
    // I2C1 is connected to the onboard IMU on Nano RP2040 Connect
    // Using 1 MHz (Fast Mode Plus). 2 MHz is out of spec but often works.
    Wire1.begin();
    Wire1.setClock(1000000);
    
    delay(10);
    
    // Verify IMU presence
    uint8_t whoami = readReg(REG_WHO_AM_I);
    if (whoami != 0x6C) {
        // Send error message in ASCII so it's readable
        Serial.print("ERROR: IMU not found. WHO_AM_I = 0x");
        Serial.println(whoami, HEX);
        while (1) delay(1000);
    }
    
    // Reset and configure
    writeReg(REG_CTRL3_C, 0x01);  // Software reset
    delay(20);
    
    // Configure for maximum ODR
    // CTRL1_XL: ODR 6.66 kHz, ±4g, LPF2 enabled
    writeReg(REG_CTRL1_XL, 0xA8);  // 1010 10 00 = 6.66kHz, ±4g
    
    // CTRL2_G: ODR 6.66 kHz, ±2000 dps
    writeReg(REG_CTRL2_G, 0xAC);   // 1010 11 00 = 6.66kHz, ±2000dps
    
    // CTRL3_C: Block data update, auto-increment
    writeReg(REG_CTRL3_C, 0x44);
    
    // CTRL4_C: Disable I2C interface = 0 (keep I2C), LPF1 bandwidth
    writeReg(REG_CTRL4_C, 0x00);
    
    // CTRL6_C: Gyro LPF1 bandwidth selection
    writeReg(REG_CTRL6_C, 0x00);
    
    // CTRL7_G: Gyro high-pass filter disabled
    writeReg(REG_CTRL7_G, 0x00);
    
    delay(50);  // Let sensor stabilize
    
    // Prepare static sync bytes
    txBuffer[0] = SYNC_BYTE_1;
    txBuffer[1] = SYNC_BYTE_2;
}

void loop() {
    // Poll status register for new data
    uint8_t status = readReg(REG_STATUS);
    
    // Wait for both accel and gyro data ready
    if ((status & (STATUS_XLDA | STATUS_GDA)) == (STATUS_XLDA | STATUS_GDA)) {
        // Burst read: 12 bytes starting at OUTX_L_G
        // Order: Gx(2), Gy(2), Gz(2), Ax(2), Ay(2), Az(2)
        burstRead(REG_OUTX_L_G, &txBuffer[2], 12);
        
        // Send as raw binary
        Serial.write(txBuffer, 14);
    }
}