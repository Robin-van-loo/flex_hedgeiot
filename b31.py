import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys
import math

class BuildingOptimizer:
    def __init__(self):
        """Initialize the building optimizer with default parameters"""
        # Core simulation parameters
        self.hours_per_day = 24
        self.forecast_days = 1  # Default forecast period
        self.day_types = []  # Will store weekday/weekend info
        
        # Building thermal parameters
        self.zones = {}  # Will store zone definitions
        self.comfort_temp_min_working = 20.0  # Min comfort temperature (°C) during working hours
        self.comfort_temp_max_working = 23.0  # Max comfort temperature (°C) during working hours
        self.comfort_temp_min_nonworking = 16.0  # Min temperature (°C) during non-working hours
        self.comfort_temp_max_nonworking = 25.0  # Max temperature (°C) during non-working hours
        
        # HVAC system parameters
        self.heat_sources = {}  # Will store heat source definitions
        self.cooling_sources = {}  # Will store cooling source definitions
        self.zone_source_mapping = {}  # Will store which heat sources serve which zones
        self.zone_cooling_mapping = {}  # Will store which cooling sources serve which zones
        
        # Energy system parameters
        self.pv_system = None  # Will store PV system parameters
        self.battery_system = None  # Will store battery system parameters
        
        # Cost parameters
        self.energy_prices = []  # Will store hourly energy prices
        
        # External conditions
        self.external_temperatures = []  # Will store external temperature forecasts
        self.solar_radiation = []  # Will store solar radiation forecasts
        
        # Define piecewise linear COP curves for heating
        self.pwl_outdoor_temp = [-15, -10, -7, -2, 2, 7, 10, 15, 20]
        self.pwl_cop_air_air_values = [2.0, 2.4, 2.6, 3.0, 3.4, 3.8, 4.0, 4.3, 4.5]
        self.pwl_cop_air_water_values = [1.9, 2.3, 2.5, 2.9, 3.2, 3.6, 3.8, 4.1, 4.4]
        
        # Define piecewise linear EER curves for cooling
        self.pwl_outdoor_temp_cooling = [20, 25, 30, 35, 40, 45]
        self.pwl_eer_air_air_values = [4.5, 4.0, 3.5, 3.0, 2.5, 2.0]  # EER decreases as outdoor temp increases
        self.pwl_eer_air_water_values = [4.3, 3.8, 3.3, 2.8, 2.3, 1.8]  # Slightly lower than air-air
        
        # COP decay factor for startup
        self.cop_decay_factor = 0.3
        
        # Gurobi environment
        self.env = gp.Env(empty=True)
        self.env.start()
        
    def define_zones(self, zones_dict):
        """
        Define building zones with thermal properties
        
        Parameters:
        -----------
        zones_dict : dict
            Dictionary with zone names as keys and zone properties as values
            Each zone should have area, volume, window_area, R, C, initial_temp
        """
        self.zones = zones_dict
        return self
    
    def define_heat_sources(self, sources_dict, cooling_sources_dict=None):
        """
        Define heat sources available in the building and optional cooling sources
        
        Parameters:
        -----------
        sources_dict : dict
            Dictionary with source names as keys and source properties as values
            Each source should have capacity, efficiency, etc.
        cooling_sources_dict : dict, optional
            Dictionary with cooling source names as keys and cooling properties as values
        """
        self.heat_sources = sources_dict
        self.cooling_sources = cooling_sources_dict or {}
        return self
    
    def map_zones_to_sources(self, mapping_dict):
        """
        Define which heat sources serve which zones and in what proportion
        
        Parameters:
        -----------
        mapping_dict : dict
            Dictionary with zone names as keys and another dict as values
            The inner dict should have source names as keys and proportion as values
        """
        self.zone_source_mapping = mapping_dict
        return self
    
    def map_zones_to_cooling_sources(self, mapping_dict):
        """
        Define which cooling sources serve which zones and in what proportion
        
        Parameters:
        -----------
        mapping_dict : dict
            Dictionary with zone names as keys and another dict as values
            The inner dict should have cooling source names as keys and proportion as values
        """
        self.zone_cooling_mapping = mapping_dict
        return self
    
    def set_cop_curves(self, outdoor_temp, air_air_cop, air_water_cop, decay_factor=0.3):
        """
        Set the piecewise linear COP curves for heat pumps
        
        Parameters:
        -----------
        outdoor_temp : list
            List of outdoor temperatures for PWL curve
        air_air_cop : list
            List of COP values for air-to-air heat pump corresponding to outdoor_temp
        air_water_cop : list
            List of COP values for air-to-water heat pump corresponding to outdoor_temp
        decay_factor : float
            Factor to decay COP during startup (default: 0.3)
        """
        self.pwl_outdoor_temp = outdoor_temp
        self.pwl_cop_air_air_values = air_air_cop
        self.pwl_cop_air_water_values = air_water_cop
        self.cop_decay_factor = decay_factor
        return self
    
    def set_cooling_cop_curves(self, outdoor_temp, air_air_eer, air_water_eer):
        """
        Set the piecewise linear EER curves for cooling equipment
        
        Parameters:
        -----------
        outdoor_temp : list
            List of outdoor temperatures for PWL curve
        air_air_eer : list
            List of EER values for air-to-air cooling corresponding to outdoor_temp
        air_water_eer : list
            List of EER values for air-to-water cooling corresponding to outdoor_temp
        """
        self.pwl_outdoor_temp_cooling = outdoor_temp
        self.pwl_eer_air_air_values = air_air_eer
        self.pwl_eer_air_water_values = air_water_eer
        return self
    
    def define_pv_system(self, capacity, efficiency, area):
        """Define the PV system parameters"""
        self.pv_system = {
            "capacity": capacity,  # kWp 
            "efficiency": efficiency,  # panel efficiency
            "area": area,  # m²
        }
        return self
    
    def define_battery_system(self, capacity, charge_eff, discharge_eff, 
                             max_charge_rate, max_discharge_rate, 
                             min_soc, max_soc, initial_soc):
        """Define the battery system parameters"""
        self.battery_system = {
            "capacity": capacity,  # kWh
            "charge_efficiency": charge_eff,
            "discharge_efficiency": discharge_eff,
            "max_charge_rate": max_charge_rate,  # kW
            "max_discharge_rate": max_discharge_rate,  # kW
            "min_soc": min_soc,  # fraction
            "max_soc": max_soc,  # fraction
            "initial_soc": initial_soc,  # fraction
        }
        return self
    
    def set_external_conditions(self, start_date, temperatures, solar_radiation, energy_prices):
        """
        Set forecasted external conditions for the optimization period
        
        Parameters:
        -----------
        start_date : datetime
            Start date and time for the simulation
        temperatures : list
            Hourly external temperatures for the simulation period
        solar_radiation : list
            Hourly solar radiation values for the simulation period
        energy_prices : list
            Hourly energy prices for the simulation period
        """
        self.start_date = start_date
        
        # Make sure all arrays are of sufficient length
        hours = self.forecast_days * self.hours_per_day
        
        # Truncate or extend arrays as needed
        self.external_temperatures = temperatures[:hours] if len(temperatures) >= hours else temperatures + [temperatures[-1]] * (hours - len(temperatures))
        self.solar_radiation = solar_radiation[:hours] if len(solar_radiation) >= hours else solar_radiation + [solar_radiation[-1]] * (hours - len(solar_radiation))
        self.energy_prices = energy_prices[:hours] if len(energy_prices) >= hours else energy_prices + [energy_prices[-1]] * (hours - len(energy_prices))
        
        # Generate day types (weekday/weekend)
        self.day_types = []
        for i in range(hours):
            current_dt = start_date + timedelta(hours=i)
            is_weekend = current_dt.weekday() >= 5
            hour_of_day = current_dt.hour
            is_working_hour = 8 <= hour_of_day < 18 and not is_weekend
            self.day_types.append({"is_weekend": is_weekend, "is_working_hour": is_working_hour})
            
        return self
    
    def calculate_pv_generation(self):
        """Calculate PV generation based on solar radiation data"""
        pv_area = self.pv_system["area"]  # PV panel area in m²
        pv_efficiency = self.pv_system["efficiency"]  # PV panel efficiency
        pv_capacity = self.pv_system["capacity"]  # kWp

        # Conversion factors from J/cm² to kWh
        J_per_cm2_to_J_per_m2 = 10_000  # 1 cm² = 0.0001 m²
        J_to_kWh = 1 / 3_600_000  # 1 kWh = 3,600,000 J

        # Convert solar radiation (J/cm²) to PV output (kWh)
        # Cap the output at the system capacity
        pv_output = []
        for q in self.solar_radiation:
            energy = q * J_per_cm2_to_J_per_m2 * J_to_kWh * pv_area * pv_efficiency
            pv_output.append(min(energy, pv_capacity))

        return pv_output
    
    def build_model(self):
        """Build the optimization model with both heating and cooling capabilities"""
        hours = self.forecast_days * self.hours_per_day
        model = gp.Model("BuildingOptimization", env=self.env)

        # Set Gurobi parameters
        model.Params.NonConvex = 2  # Handle non-convex constraints
        model.Params.NumericFocus = 3  # Focus on numerical stability
        model.Params.TimeLimit = 300  # 5 minute time limit

        # Calculate PV generation
        pv_generation = self.calculate_pv_generation()

        # 1. Create variables

        # External temperature variables (for PWL constraints)
        T_out = model.addVars(hours, lb=-50, ub=50, name="ExternalTemp")

        # Temperature variables for each zone
        zone_temps = {}
        for zone_name in self.zones:
            zone_temps[zone_name] = model.addVars(hours, lb=0, name=f"Temp_{zone_name}")

        # Heat input variables for each zone and heat source
        zone_heat_input = {}
        for zone_name in self.zones:
            zone_heat_input[zone_name] = {}
            for source_name in self.zone_source_mapping.get(zone_name, {}):
                zone_heat_input[zone_name][source_name] = model.addVars(hours, lb=0, name=f"Heat_{zone_name}_{source_name}")

        # Cooling input variables for each zone and cooling source
        zone_cool_input = {}
        for zone_name in self.zones:
            zone_cool_input[zone_name] = {}
            for source_name in self.zone_cooling_mapping.get(zone_name, {}):
                zone_cool_input[zone_name][source_name] = model.addVars(hours, lb=0, name=f"Cool_{zone_name}_{source_name}")

        # Heat source variables
        source_power = {}  # Thermal output (kW)
        source_on = {}     # Binary on/off variables
        source_energy = {} # Electrical input (kW)
        source_cop = {}    # COP variables
        source_startup = {}  # Startup indicator variables

        for source_name, source_data in self.heat_sources.items():
            source_power[source_name] = model.addVars(hours, lb=0, ub=source_data["capacity"], name=f"Power_{source_name}")
            source_on[source_name] = model.addVars(hours, vtype=GRB.BINARY, name=f"On_{source_name}")
            source_energy[source_name] = model.addVars(hours, lb=0, name=f"Energy_{source_name}")

            # Special variables for heat pumps with COP modeling
            if source_name in ["AirToAir_HP", "AirToWater_HP"]:
                source_cop[source_name] = model.addVars(hours, lb=1.0, name=f"COP_{source_name}")
                source_startup[source_name] = model.addVars(hours, vtype=GRB.BINARY, name=f"Startup_{source_name}")

        # Cooling source variables
        cooling_power = {}  # Cooling capacity (kW)
        cooling_on = {}     # Binary on/off variables
        cooling_energy = {} # Electrical input (kW)
        cooling_cop = {}    # Cooling COP (EER) variables
        cooling_startup = {}  # Startup indicator variables

        for source_name, source_data in self.cooling_sources.items():
            cooling_power[source_name] = model.addVars(hours, lb=0, ub=source_data["capacity"], name=f"CoolPower_{source_name}")
            cooling_on[source_name] = model.addVars(hours, vtype=GRB.BINARY, name=f"CoolOn_{source_name}")
            cooling_energy[source_name] = model.addVars(hours, lb=0, name=f"CoolEnergy_{source_name}")

            # Special variables for heat pumps with EER modeling
            if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
                cooling_cop[source_name] = model.addVars(hours, lb=1.0, name=f"EER_{source_name}")
                cooling_startup[source_name] = model.addVars(hours, vtype=GRB.BINARY, name=f"CoolStartup_{source_name}")

        # Heating/cooling mode selection - prevent simultaneous heating and cooling in the same zone
        # This is a binary variable where 1 = heating mode, 0 = cooling mode
        zone_heating_mode = {}
        for zone_name in self.zones:
            if zone_name in self.zone_source_mapping and zone_name in self.zone_cooling_mapping:
                zone_heating_mode[zone_name] = model.addVars(hours, vtype=GRB.BINARY, name=f"HeatingMode_{zone_name}")

        # Total heating and cooling power
        total_heating_power = model.addVars(hours, lb=0, name="TotalHeatingPower")
        total_cooling_power = model.addVars(hours, lb=0, name="TotalCoolingPower")

        # PV-related variables
        pv_self_consumed = model.addVars(hours, lb=0, name="PV_SelfConsumed")
        grid_imported = model.addVars(hours, lb=0, name="Grid_Imported")
        pv_waste = model.addVars(hours, lb=0, name="PV_Waste")

        # Battery-related variables (handle zero capacity)
        if self.battery_system["capacity"] > 0:
            batt_soc = model.addVars(hours+1, 
                                   lb=self.battery_system["min_soc"] * self.battery_system["capacity"], 
                                   ub=self.battery_system["max_soc"] * self.battery_system["capacity"], 
                                   name="Battery_SOC")
            batt_charge = model.addVars(hours, lb=0, 
                                      ub=self.battery_system["max_charge_rate"], 
                                      name="Battery_Charge")
            batt_discharge = model.addVars(hours, lb=0, 
                                         ub=self.battery_system["max_discharge_rate"], 
                                         name="Battery_Discharge")
            batt_charge_grid = model.addVars(hours, lb=0, name="Battery_Charge_Grid")
            batt_charge_pv = model.addVars(hours, lb=0, name="Battery_Charge_PV")
            batt_mode = model.addVars(hours, vtype=GRB.BINARY, name="Battery_Mode")
        else:
            # Create dummy variables with zero bounds for no battery case
            batt_soc = model.addVars(hours+1, lb=0, ub=0, name="Battery_SOC")
            batt_charge = model.addVars(hours, lb=0, ub=0, name="Battery_Charge")
            batt_discharge = model.addVars(hours, lb=0, ub=0, name="Battery_Discharge")
            batt_charge_grid = model.addVars(hours, lb=0, ub=0, name="Battery_Charge_Grid")
            batt_charge_pv = model.addVars(hours, lb=0, ub=0, name="Battery_Charge_PV")
            batt_mode = model.addVars(hours, lb=0, ub=0, name="Battery_Mode")  # Not binary when zero

        # 2. Set initial conditions

        # Set initial zone temperatures
        for zone_name, zone_data in self.zones.items():
            model.addConstr(zone_temps[zone_name][0] == zone_data["initial_temp"])

        # Set initial battery SOC (handle zero capacity)
        if self.battery_system["capacity"] > 0:
            model.addConstr(batt_soc[0] == self.battery_system["initial_soc"] * self.battery_system["capacity"])
        else:
            model.addConstr(batt_soc[0] == 0)

        # 3. Add constraints

        # External temperature constraint (set the variable equal to the input data)
        for i in range(hours):
            model.addConstr(T_out[i] == self.external_temperatures[i])

        # COP constraints using piecewise linear functions for heating and cooling
        for i in range(hours):
            # Air-to-air heat pump COP for heating
            if "AirToAir_HP" in self.heat_sources:
                # Add PWL constraint for base COP
                model.addGenConstrPWL(
                    T_out[i], 
                    source_cop["AirToAir_HP"][i], 
                    self.pwl_outdoor_temp, 
                    self.pwl_cop_air_air_values
                )

            # Air-to-water heat pump COP for heating
            if "AirToWater_HP" in self.heat_sources:
                # Add PWL constraint for base COP
                model.addGenConstrPWL(
                    T_out[i], 
                    source_cop["AirToWater_HP"][i], 
                    self.pwl_outdoor_temp, 
                    self.pwl_cop_air_water_values
                )
                
            # Air-to-air heat pump EER for cooling
            if "AirToAir_AC" in self.cooling_sources:
                # Add PWL constraint for cooling EER
                model.addGenConstrPWL(
                    T_out[i], 
                    cooling_cop["AirToAir_AC"][i], 
                    self.pwl_outdoor_temp_cooling, 
                    self.pwl_eer_air_air_values
                )
                
            # Chiller EER for cooling
            if "AirToWater_Chiller" in self.cooling_sources:
                # Add PWL constraint for cooling EER
                model.addGenConstrPWL(
                    T_out[i], 
                    cooling_cop["AirToWater_Chiller"][i], 
                    self.pwl_outdoor_temp_cooling, 
                    self.pwl_eer_air_water_values
                )

        # Startup detection for heating
        for source_name in ["AirToAir_HP", "AirToWater_HP"]:
            if source_name in self.heat_sources:
                for i in range(hours):
                    if i > 0:
                        # Startup occurs when source is on now but was off in previous hour
                        model.addConstr(source_startup[source_name][i] >= source_on[source_name][i] - source_on[source_name][i-1])
                    else:
                        # For first hour, startup occurs if source is on
                        model.addConstr(source_startup[source_name][i] >= source_on[source_name][i])
                        
        # Startup detection for cooling
        for source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
            if source_name in self.cooling_sources:
                for i in range(hours):
                    if i > 0:
                        # Startup occurs when source is on now but was off in previous hour
                        model.addConstr(cooling_startup[source_name][i] >= cooling_on[source_name][i] - cooling_on[source_name][i-1])
                    else:
                        # For first hour, startup occurs if source is on
                        model.addConstr(cooling_startup[source_name][i] >= cooling_on[source_name][i])

        # Battery constraints (handle zero capacity)
        for i in range(hours):
            if self.battery_system["capacity"] > 0:
                # Can't charge and discharge simultaneously
                model.addConstr(batt_charge[i] <= batt_mode[i] * self.battery_system["max_charge_rate"])
                model.addConstr(batt_discharge[i] <= (1 - batt_mode[i]) * self.battery_system["max_discharge_rate"])
                model.addConstr(batt_charge[i] == batt_charge_grid[i] + batt_charge_pv[i])

                # Battery SOC evolution
                model.addConstr(
                    batt_soc[i+1] == batt_soc[i] + 
                    self.battery_system["charge_efficiency"] * batt_charge[i] - 
                    batt_discharge[i] / self.battery_system["discharge_efficiency"]
                )
            else:
                # For no battery case, all battery variables stay at zero
                model.addConstr(batt_charge[i] == 0)
                model.addConstr(batt_discharge[i] == 0)
                model.addConstr(batt_charge_grid[i] == 0)
                model.addConstr(batt_charge_pv[i] == 0)
                model.addConstr(batt_soc[i+1] == 0)

        # Heat source constraints
        for i in range(hours):
            for source_name, source_data in self.heat_sources.items():
                # Link source power to energy consumption
                if source_name in ["AirToAir_HP", "AirToWater_HP"]:
                    # For heat pumps, use the COP with startup penalty
                    # Calculate effective COP after considering startup penalty
                    effective_cop = source_cop[source_name][i] - self.cop_decay_factor * source_startup[source_name][i]

                    # Link thermal power output to electrical energy input through COP
                    model.addConstr(source_power[source_name][i] == source_energy[source_name][i] * effective_cop)
                else:
                    # For non-heat pump sources like boilers, use fixed efficiency
                    model.addConstr(source_power[source_name][i] == source_energy[source_name][i] * source_data["efficiency"])

                # Heat source on/off logic
                model.addConstr(source_power[source_name][i] <= source_data["capacity"] * source_on[source_name][i])
                model.addConstr(source_power[source_name][i] >= 0.1 * source_on[source_name][i])  # Minimum power if on
                
        # Cooling source constraints
        for i in range(hours):
            for source_name, source_data in self.cooling_sources.items():
                # Link source cooling power to energy consumption
                if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
                    # For cooling systems, use the EER with startup penalty
                    # Calculate effective EER after considering startup penalty
                    effective_eer = cooling_cop[source_name][i] - self.cop_decay_factor * cooling_startup[source_name][i]

                    # Link cooling power output to electrical energy input through EER
                    model.addConstr(cooling_power[source_name][i] == cooling_energy[source_name][i] * effective_eer)
                else:
                    # For other cooling sources, use fixed efficiency
                    model.addConstr(cooling_power[source_name][i] == cooling_energy[source_name][i] * source_data["efficiency"])

                # Cooling source on/off logic
                model.addConstr(cooling_power[source_name][i] <= source_data["capacity"] * cooling_on[source_name][i])
                model.addConstr(cooling_power[source_name][i] >= 0.1 * cooling_on[source_name][i])  # Minimum power if on

        # Prevent simultaneous heating and cooling in the same zone
        for i in range(hours):
            for zone_name in self.zones:
                if zone_name in zone_heating_mode:
                    # Sum of heating inputs for this zone
                    if zone_name in self.zone_source_mapping:
                        for source_name in self.zone_source_mapping[zone_name]:
                            # Heating can only be on in heating mode (zone_heating_mode = 1)
                            model.addConstr(zone_heat_input[zone_name][source_name][i] <= 
                                          1000 * zone_heating_mode[zone_name][i])  # Big M constraint
                    
                    # Sum of cooling inputs for this zone
                    if zone_name in self.zone_cooling_mapping:
                        for source_name in self.zone_cooling_mapping[zone_name]:
                            # Cooling can only be on in cooling mode (zone_heating_mode = 0)
                            model.addConstr(zone_cool_input[zone_name][source_name][i] <= 
                                          1000 * (1 - zone_heating_mode[zone_name][i]))  # Big M constraint

        # Calculate total heating and cooling power
        for i in range(hours):
            # Total heating power
            model.addConstr(total_heating_power[i] == quicksum(source_power[source_name][i] 
                                                          for source_name in self.heat_sources))
            
            # Total cooling power
            model.addConstr(total_cooling_power[i] == quicksum(cooling_power[source_name][i] 
                                                          for source_name in self.cooling_sources))

            # Distribute heat source output to zones based on mapping
            for source_name in self.heat_sources:
                model.addConstr(source_power[source_name][i] == 
                              quicksum(zone_heat_input[zone_name][source_name][i] 
                                     for zone_name in self.zones if source_name in self.zone_source_mapping.get(zone_name, {})))
                                     
            # Distribute cooling source output to zones based on mapping
            for source_name in self.cooling_sources:
                model.addConstr(cooling_power[source_name][i] == 
                              quicksum(zone_cool_input[zone_name][source_name][i] 
                                     for zone_name in self.zones if source_name in self.zone_cooling_mapping.get(zone_name, {})))

            # Calculate zone temperatures based on thermal dynamics
            if i > 0:
                for zone_name, zone_data in self.zones.items():
                    # Solar gain for this zone
                    window_area = zone_data["window_area"]
                    solar_gain = self.solar_radiation[i] * 0.7 * window_area / 1000  # Convert to kW

                    # Sum of heat inputs from all sources for this zone
                    zone_total_heat = quicksum(zone_heat_input[zone_name][source_name][i] 
                                            for source_name in self.zone_source_mapping.get(zone_name, {}))
                    
                    # Sum of cooling inputs from all sources for this zone
                    zone_total_cool = quicksum(zone_cool_input[zone_name][source_name][i] 
                                            for source_name in self.zone_cooling_mapping.get(zone_name, {}))

                    # Thermal dynamics equation with both heating and cooling
                    model.addConstr(
                        zone_temps[zone_name][i] == zone_temps[zone_name][i-1] +
                        (1 / zone_data["C"]) * (
                            (self.external_temperatures[i] - zone_temps[zone_name][i-1]) / zone_data["R"] +
                            zone_total_heat -  # Positive contribution (heating)
                            zone_total_cool +  # Negative contribution (cooling)
                            solar_gain
                        )
                    )

            # Comfort constraints based on working/non-working hours
            is_working_hour = self.day_types[i]["is_working_hour"]
            for zone_name in self.zones:
                if is_working_hour:
                    model.addConstr(zone_temps[zone_name][i] >= self.comfort_temp_min_working)
                    model.addConstr(zone_temps[zone_name][i] <= self.comfort_temp_max_working)
                else:
                    model.addConstr(zone_temps[zone_name][i] >= self.comfort_temp_min_nonworking)
                    model.addConstr(zone_temps[zone_name][i] <= self.comfort_temp_max_nonworking)

        # Energy balance constraints WITH PV PRIORITY
        for i in range(hours):
            # Calculate total electrical demand (heating + cooling)
            electrical_demand = quicksum(source_energy[source_name][i] for source_name in self.heat_sources) + \
                           quicksum(cooling_energy[source_name][i] for source_name in self.cooling_sources)

            # CONSTRAINT 1: Energy balance - total supply must equal total demand
            model.addConstr(
            pv_self_consumed[i] + grid_imported[i] + batt_discharge[i] == 
            electrical_demand + batt_charge_grid[i]
            )

            # CONSTRAINT 2: PV allocation balance - all PV must be accounted for
            model.addConstr(
                pv_self_consumed[i] + batt_charge_pv[i] + pv_waste[i] == pv_generation[i]
            )

            # CONSTRAINT 3: PV self-consumption limits
            model.addConstr(pv_self_consumed[i] <= electrical_demand)
            model.addConstr(pv_self_consumed[i] <= pv_generation[i])

            # CONSTRAINT 4: Battery PV charging only from surplus
            model.addConstr(batt_charge_pv[i] <= pv_generation[i] - pv_self_consumed[i])

            # CONSTRAINT 5: Total battery charging
            model.addConstr(batt_charge[i] == batt_charge_grid[i] + batt_charge_pv[i])
    
        # Create temperature violation variables and enhanced comfort constraints
        temp_comfort_violation = {}
        for zone_name in self.zones:
            temp_comfort_violation[zone_name] = model.addVars(hours, lb=0, name=f"Temp_Violation_{zone_name}")

        # Set objective function with very high penalty for temperature violations
        comfort_penalty_weight = 10000000  # 10 million
        # And modify it to include a penalty for low final battery SOC:
        final_soc = batt_soc[hours]  # SOC at the end of the optimization period
        min_soc = self.battery_system["min_soc"] * self.battery_system["capacity"]
        max_soc = self.battery_system["max_soc"] * self.battery_system["capacity"]
        target_soc = min_soc + 0.3 * (max_soc - min_soc)  # Target ending at 30% between min and max

        # Add penalty term to discourage draining battery completely
        battery_drain_penalty = 50  # Adjust this weight as needed
        model.setObjective(
            quicksum(grid_imported[i] * self.energy_prices[i] for i in range(hours)) +
            0.1 * quicksum(pv_waste[i] for i in range(hours)) +
            comfort_penalty_weight * quicksum(temp_comfort_violation[zone_name][i] 
                                                for zone_name in self.zones 
                                                for i in range(hours)) +
            battery_drain_penalty * (target_soc - final_soc),  # Penalty increases as final SOC falls below target
            GRB.MINIMIZE
        )

        # Store model and variables
        self.model = model
        
        self.variables = {
            "zone_temps": zone_temps,
            "zone_heat_input": zone_heat_input,
            "zone_cool_input": zone_cool_input,
            "source_power": source_power,
            "source_on": source_on,
            "source_energy": source_energy,
            "source_cop": source_cop,
            "source_startup": source_startup,
            "cooling_power": cooling_power,
            "cooling_on": cooling_on,
            "cooling_energy": cooling_energy,
            "cooling_cop": cooling_cop,
            "cooling_startup": cooling_startup,
            "total_heating_power": total_heating_power,
            "total_cooling_power": total_cooling_power,
            "pv_self_consumed": pv_self_consumed,
            "grid_imported": grid_imported,
            "pv_waste": pv_waste,
            "batt_soc": batt_soc,
            "batt_charge": batt_charge,
            "batt_discharge": batt_discharge,
            "batt_charge_grid": batt_charge_grid,
            "batt_charge_pv": batt_charge_pv,
            "T_out": T_out,
            "temp_comfort_violation": temp_comfort_violation,
            "zone_heating_mode": zone_heating_mode
        }
    
        return self
        
    def optimize(self):
        """Run the optimization"""
        if not hasattr(self, 'model'):
            self.build_model()
            
        # Optimize the model
        self.model.optimize()
        
        # Check status
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT or self.model.status == GRB.SUBOPTIMAL:
            print(f"Optimization completed with status: {self.model.status}")
            print(f"Objective value: {self.model.ObjVal:.2f}")
            return True
        else:
            print(f"Optimization failed with status: {self.model.status}")
            return False
        
    def extract_results(self):
        """Extract and structure the optimization results including cooling data"""
        hours = self.forecast_days * self.hours_per_day
        
        # Create result structure
        results = {
            "temperatures": {
                "external": self.external_temperatures,
                "zones": {}
            },
            "heating": {
                "total_power": [self.variables["total_heating_power"][i].x for i in range(hours)],
                "sources": {}
            },
            "cooling": {
                "total_power": [self.variables["total_cooling_power"][i].x for i in range(hours)],
                "sources": {}
            },
            "energy": {
                "prices": self.energy_prices,
                "grid_imported": [self.variables["grid_imported"][i].x for i in range(hours)],
                "pv": {
                    "generation": self.calculate_pv_generation(),
                    "self_consumed": [self.variables["pv_self_consumed"][i].x for i in range(hours)],
                    "waste": [self.variables["pv_waste"][i].x for i in range(hours)]
                },
                "battery": {
                    "soc": [self.variables["batt_soc"][i].x for i in range(hours+1)],
                    "charge": [self.variables["batt_charge"][i].x for i in range(hours)],
                    "discharge": [self.variables["batt_discharge"][i].x for i in range(hours)],
                    "charge_grid": [self.variables["batt_charge_grid"][i].x for i in range(hours)],
                    "charge_pv": [self.variables["batt_charge_pv"][i].x for i in range(hours)]
                }
            },
            "timestamps": [
                (self.start_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
                for i in range(hours)
            ],
            "is_working_hour": [self.day_types[i]["is_working_hour"] for i in range(hours)]
        }
        
        # Extract zone temperatures and mode
        for zone_name in self.zones:
            results["temperatures"]["zones"][zone_name] = [
                self.variables["zone_temps"][zone_name][i].x for i in range(hours)
            ]
            
        # Extract heating/cooling mode if available
        if "zone_heating_mode" in self.variables:
            results["zone_modes"] = {}
            for zone_name in self.variables["zone_heating_mode"]:
                results["zone_modes"][zone_name] = [
                    self.variables["zone_heating_mode"][zone_name][i].x for i in range(hours)
                ]
        
        # Extract heat source data with COP values
        for source_name in self.heat_sources:
            source_data = {
                "power": [self.variables["source_power"][source_name][i].x for i in range(hours)],
                "energy": [self.variables["source_energy"][source_name][i].x for i in range(hours)],
                "on": [self.variables["source_on"][source_name][i].x for i in range(hours)]
            }
            
            # Add COP and startup data for heat pumps
            if source_name in ["AirToAir_HP", "AirToWater_HP"]:
                source_data["cop"] = [self.variables["source_cop"][source_name][i].x for i in range(hours)]
                source_data["startup"] = [self.variables["source_startup"][source_name][i].x for i in range(hours)]
                
                # Calculate effective COP (after startup penalty)
                effective_cop = []
                for i in range(hours):
                    cop = source_data["cop"][i]
                    startup = source_data["startup"][i]
                    eff_cop = cop - self.cop_decay_factor * startup
                    effective_cop.append(eff_cop)
                
                source_data["effective_cop"] = effective_cop
            
            results["heating"]["sources"][source_name] = source_data
        
        # Extract cooling source data with EER values
        for source_name in self.cooling_sources:
            source_data = {
                "power": [self.variables["cooling_power"][source_name][i].x for i in range(hours)],
                "energy": [self.variables["cooling_energy"][source_name][i].x for i in range(hours)],
                "on": [self.variables["cooling_on"][source_name][i].x for i in range(hours)]
            }
            
            # Add EER and startup data for cooling systems
            if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
                source_data["eer"] = [self.variables["cooling_cop"][source_name][i].x for i in range(hours)]
                source_data["startup"] = [self.variables["cooling_startup"][source_name][i].x for i in range(hours)]
                
                # Calculate effective EER (after startup penalty)
                effective_eer = []
                for i in range(hours):
                    eer = source_data["eer"][i]
                    startup = source_data["startup"][i]
                    eff_eer = eer - self.cop_decay_factor * startup
                    effective_eer.append(eff_eer)
                
                source_data["effective_eer"] = effective_eer
            
            results["cooling"]["sources"][source_name] = source_data
        
        # Calculate total cost
        total_cost = sum((results["energy"]["grid_imported"][i] * results["energy"]["prices"][i] / 1000)
                        for i in range(hours))
        results["total_cost"] = total_cost
        
        # Calculate PV self-consumption rate
        total_pv = sum(results["energy"]["pv"]["generation"])
        pv_consumed = sum(results["energy"]["pv"]["self_consumed"]) + sum(results["energy"]["battery"]["charge_pv"])
        results["energy"]["pv"]["self_consumption_rate"] = pv_consumed / total_pv if total_pv > 0 else 0
        
        return results
    
    def save_results_to_csv(self, results, filename="building_optimization_results.csv"):
        """Save the optimization results to a CSV file"""
        hours = self.forecast_days * self.hours_per_day
        
        # Create a dataframe
        data = {
            "timestamp": results["timestamps"],
            "is_working_hour": results["is_working_hour"],
            "external_temp": results["temperatures"]["external"],
            "energy_price": results["energy"]["prices"],
            "total_heating_power": results["heating"]["total_power"],
            "total_cooling_power": results["cooling"]["total_power"],
            "grid_imported": results["energy"]["grid_imported"],
            "pv_generation": results["energy"]["pv"]["generation"],
            "pv_self_consumed": results["energy"]["pv"]["self_consumed"],
            "pv_waste": results["energy"]["pv"]["waste"],
            "battery_soc": results["energy"]["battery"]["soc"][:-1],  # Remove last element (hour+1)
            "battery_charge": results["energy"]["battery"]["charge"],
            "battery_discharge": results["energy"]["battery"]["discharge"],
            "battery_charge_grid": results["energy"]["battery"]["charge_grid"],
            "battery_charge_pv": results["energy"]["battery"]["charge_pv"]
        }
        
        # Add zone temperatures
        for zone_name in self.zones:
            data[f"temp_{zone_name}"] = results["temperatures"]["zones"][zone_name]
        
        # Add zone heating/cooling mode if available
        if "zone_modes" in results:
            for zone_name in results["zone_modes"]:
                data[f"heating_mode_{zone_name}"] = results["zone_modes"][zone_name]
        
        # Add heat source data
        for source_name in self.heat_sources:
            data[f"power_{source_name}"] = results["heating"]["sources"][source_name]["power"]
            data[f"energy_{source_name}"] = results["heating"]["sources"][source_name]["energy"]
            data[f"on_{source_name}"] = results["heating"]["sources"][source_name]["on"]
            
            # Add COP data for heat pumps
            if source_name in ["AirToAir_HP", "AirToWater_HP"]:
                data[f"cop_{source_name}"] = results["heating"]["sources"][source_name]["cop"]
                data[f"effective_cop_{source_name}"] = results["heating"]["sources"][source_name]["effective_cop"]
                data[f"startup_{source_name}"] = results["heating"]["sources"][source_name]["startup"]
        
        # Add cooling source data
        for source_name in self.cooling_sources:
            data[f"power_{source_name}"] = results["cooling"]["sources"][source_name]["power"]
            data[f"energy_{source_name}"] = results["cooling"]["sources"][source_name]["energy"]
            data[f"on_{source_name}"] = results["cooling"]["sources"][source_name]["on"]
            
            # Add EER data for cooling systems
            if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
                data[f"eer_{source_name}"] = results["cooling"]["sources"][source_name]["eer"]
                data[f"effective_eer_{source_name}"] = results["cooling"]["sources"][source_name]["effective_eer"]
                data[f"startup_{source_name}"] = results["cooling"]["sources"][source_name]["startup"]
        
        # Create and save dataframe
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

# Function to load or generate data
def load_or_generate_data(days_to_simulate=3, start_date=datetime(2023, 7, 15, 0, 0)):
    """
    Load data from CSV files if available, otherwise generate synthetic data
    
    Parameters:
    -----------
    days_to_simulate : int
        Number of days to include in the simulation
    start_date : datetime
        Start date and time for the simulation
    
    Returns:
    --------
    tuple
        (temperatures, solar_radiation, energy_prices) for the simulation period
    """
    # Check if CSV files exist
    required_files = ["day_ahead_prices.csv", "formatted_temperature_forecast.csv", "Cleaned_PV_Data.csv"]
    all_files_exist = all(os.path.exists(file) for file in required_files)
    
    if all_files_exist:
        try:
            # Try to load data from CSV files
            print("Loading data from CSV files...")
            temperatures, solar_radiation, energy_prices = load_data_from_csv(days_to_simulate, start_date)
            print("Successfully loaded data from CSV files")
            return temperatures, solar_radiation, energy_prices
        except Exception as e:
            print(f"Error loading data from CSV: {str(e)}")
            print("Falling back to synthetic data")
    else:
        print("One or more required CSV files not found. Using synthetic data instead.")
    
    # Generate synthetic data
    return generate_synthetic_data(days_to_simulate, start_date)

def load_data_from_csv(days_to_simulate=3, start_date=datetime(2023, 7, 15, 0, 0)):
    """Load data from CSV files"""
    # Load day-ahead prices
    dah_df = pd.read_csv("day_ahead_prices.csv")
    dah_df["hour"] = dah_df["Hour"] - 1  # Convert 1-24 to 0-23
    dah_df["datetime"] = dah_df.apply(
        lambda row: f"{row['Day-Month']} {int(row['hour']):02d}:00", axis=1
    )
    dah_df["price"] = dah_df["Average Day-ahead Price (EUR/MWh)"]
    dah_price_map = dict(zip(dah_df["datetime"], dah_df["price"]))

    # Load temperature forecast
    temp_df = pd.read_csv("formatted_temperature_forecast.csv")
    temp_df["datetime"] = temp_df.apply(
        lambda row: f"{row['Day-Month']} {int(row['Hour']):02d}:00", axis=1
    )
    temp_map = dict(zip(temp_df["datetime"], temp_df["Temperature"]))

    # Load solar radiation data
    solar_df = pd.read_csv("Cleaned_PV_Data.csv")
    solar_df["Day-Month"] = solar_df["Day-Month"].astype(str).str.zfill(4)
    solar_df["hour"] = solar_df["Hour"].astype(int) - 1
    solar_df["datetime"] = solar_df.apply(
        lambda row: f"{row['Day-Month'][:2]}-{row['Day-Month'][2:]} {row['hour']:02d}:00", axis=1
    )
    solar_map = dict(zip(solar_df["datetime"], solar_df["Q_value"]))
    
    # Extract data for the requested simulation period
    hours = 24 * days_to_simulate
    temperatures = []
    solar_radiation = []
    energy_prices = []
    
    for h in range(hours):
        dt = start_date + timedelta(hours=h)
        key = dt.strftime("%m-%d %H:00")
        
        # Get temperature (default to 20°C if missing)
        temp = temp_map.get(key, 20)
        temperatures.append(temp)
        
        # Get solar radiation (default to 0 if missing)
        solar = solar_map.get(key, 0)
        solar_radiation.append(solar)
        
        # Get energy price (default to 100 EUR/MWh if missing)
        price = dah_price_map.get(key, 100)
        energy_prices.append(price)
    
    return temperatures, solar_radiation, energy_prices

def generate_synthetic_data(days, start_date):
    """Generate synthetic data for simulation"""
    print("Generating synthetic data for simulation...")
    hours = days * 24
    
    # Generate temperature data (summer pattern with day/night variations)
    is_summer = 4 <= start_date.month <= 9
    base_temp = 25 if is_summer else 5  # Base temperature depends on season
    
    temperatures = []
    for h in range(hours):
        current_dt = start_date + timedelta(hours=h)
        hour_of_day = current_dt.hour
        # Day/night variation: -5°C at night, +5°C during day
        day_factor = -5 * math.cos(2 * math.pi * hour_of_day / 24)
        # Add some random variation
        random_factor = np.random.normal(0, 1)
        temp = base_temp + day_factor + random_factor
        temperatures.append(temp)
    
    # Generate solar radiation data (zero at night, peak at noon)
    solar_radiation = []
    for h in range(hours):
        current_dt = start_date + timedelta(hours=h)
        hour_of_day = current_dt.hour
        if 6 <= hour_of_day <= 20:  # Daylight hours
            # Peak at noon, zero at sunrise/sunset
            solar_factor = math.sin(math.pi * (hour_of_day - 6) / 14)
            # Reduced solar in winter
            if not is_summer:
                solar_factor *= 0.4
            radiation = max(0, 800 * solar_factor + np.random.normal(0, 30))
        else:
            radiation = 0  # Night time
        solar_radiation.append(radiation)
    
    # Generate energy prices (higher during peak hours)
    energy_prices = []
    for h in range(hours):
        current_dt = start_date + timedelta(hours=h)
        hour_of_day = current_dt.hour
        is_weekend = current_dt.weekday() >= 5
        
        # Base price
        base_price = 70
        
        # Peak hours premium (8-20 on weekdays)
        peak_premium = 0
        if not is_weekend and 8 <= hour_of_day <= 20:
            peak_premium = 30 * math.sin(math.pi * (hour_of_day - 8) / 12)
        
        # Add some random variation
        random_factor = np.random.normal(0, 5)
        
        price = base_price + peak_premium + random_factor
        energy_prices.append(max(0, price))
    
    print("Synthetic data generated successfully")
    return temperatures, solar_radiation, energy_prices

def run_building_optimization_with_cooling(days=3, output_dir="./results", 
                                          start_date=datetime(2023, 7, 15, 0, 0),
                                          rolling_horizon=True, forecast_hours=24,
                                          zones=None, heat_sources=None, cooling_sources=None,
                                          zone_source_mapping=None, zone_cooling_mapping=None,
                                          battery_capacity=200, battery_charge_rate=60, 
                                          battery_discharge_rate=90, battery_initial_soc=100):
    """Run the complete building optimization model with cooling for a summer period"""
    print(f"Running building optimization model for {days} days starting from {start_date.strftime('%Y-%m-%d')}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use provided building data if available, otherwise generate defaults
    if zones is None or heat_sources is None or cooling_sources is None or zone_source_mapping is None or zone_cooling_mapping is None:
        zones, heat_sources, cooling_sources, zone_source_mapping, zone_cooling_mapping = generate_building_data_with_cooling()
        print("Using default building configuration")
    else:
        print("Using custom building configuration")
    
    # Load real data from CSV files
    temperatures, solar_radiation, energy_prices = load_or_generate_data(days, start_date)
    print("Successfully loaded data from CSV files")
    
    if rolling_horizon:
        # Define battery capacity from parameters
        # State tracking for rolling horizon
        current_state = {
            "battery_soc": battery_initial_soc,  # Initial battery SOC from parameter
            "zone_temps": {zone: zones[zone]["initial_temp"] for zone in zones}
        }
        
        # Storage for compiled results
        compiled_results = initialize_empty_results(days*24, zones, heat_sources, cooling_sources)
        
        # Daily optimization approach
        total_days = days
        
        # Track battery SOC across days for debugging
        battery_soc_by_day = []
        
        for day in range(total_days):
            # Record starting SOC for tracking
            battery_soc_by_day.append(current_state["battery_soc"])
            
            # Start time for this day's optimization
            day_start_time = start_date + timedelta(days=day)
            day_start_hour = day * 24
            print(f"\nOptimizing for day {day+1} ({day_start_time.strftime('%Y-%m-%d')})...")
            print(f"Starting battery SOC: {current_state['battery_soc']:.1f} kWh ({current_state['battery_soc']/battery_capacity*100:.1f}%)")
            
            # Extract 24 hours of data for this day
            day_end_hour = min(day_start_hour + 24, days * 24)  # Cap at end of simulation
            day_temps = temperatures[day_start_hour:day_end_hour]
            day_solar = solar_radiation[day_start_hour:day_end_hour]
            day_prices = energy_prices[day_start_hour:day_end_hour]
            
            # Create optimizer for this day
            optimizer = BuildingOptimizer()
            optimizer.forecast_days = 1  # Always one day at a time
            
            # Update initial temperatures based on current state
            for zone_name in zones:
                zones[zone_name]["initial_temp"] = current_state["zone_temps"][zone_name]
            
            # Configure optimizer with current state and forecast
            # IMPORTANT: Set initial_soc as a fraction of capacity
            initial_soc_fraction = current_state["battery_soc"] / battery_capacity
            
            optimizer.define_zones(zones) \
                    .define_heat_sources(heat_sources, cooling_sources) \
                    .map_zones_to_sources(zone_source_mapping) \
                    .map_zones_to_cooling_sources(zone_cooling_mapping) \
                    .define_pv_system(capacity=150, efficiency=0.18, area=300) \
                    .define_battery_system(
                        capacity=battery_capacity, 
                        charge_eff=0.95, 
                        discharge_eff=0.95,
                        max_charge_rate=battery_charge_rate,
                        max_discharge_rate=battery_discharge_rate,
                        min_soc=0.1,  # Allow minimum 10% SOC
                        max_soc=0.9,  # Allow maximum 90% SOC  
                        initial_soc=initial_soc_fraction  # Use the carried over SOC
                    ) \
                    .set_external_conditions(day_start_time, day_temps, day_solar, day_prices) \
                    .set_cooling_cop_curves(
                        outdoor_temp=[20, 25, 30, 35, 40, 45],
                        air_air_eer=[4.5, 4.0, 3.5, 3.0, 2.5, 2.0],
                        air_water_eer=[4.3, 3.8, 3.3, 2.8, 2.3, 1.8]
                    )
            
            # Build and optimize the model
            optimizer.build_model()
            success = optimizer.optimize()
            
            if success:
                # Extract results for the day
                day_results = optimizer.extract_results()
                
                # Store all 24 hours for this day
                for hour in range(24):
                    if hour < len(day_results["timestamps"]):  # Make sure we have data for this hour
                        # Update hour index in compiled results
                        hour_index = day_start_hour + hour
                        
                        # Store results for this hour
                        store_hour_results(compiled_results, day_results, hour, hour_index)
                
                # Update current state with end-of-day values for next day
                # IMPORTANT: This follows the same pattern as zone temperatures
                
                # Get the last battery SOC for this day
                soc_array = day_results["energy"]["battery"]["soc"]
                
                if len(soc_array) > 24:
                    # Use the 24th index (end of day)
                    current_state["battery_soc"] = soc_array[24]
                else:
                    # If fewer than 25 entries, use the last available
                    current_state["battery_soc"] = soc_array[-1]
                
                # Update zone temperatures (following this same pattern)
                for zone_name in zones:
                    # Use last hour's temperature as initial for next day
                    last_hour = min(23, len(day_results["temperatures"]["zones"][zone_name])-1)
                    current_state["zone_temps"][zone_name] = day_results["temperatures"]["zones"][zone_name][last_hour]
                
                # Print status update
                print(f"Day {day+1} optimized successfully")
                print(f"Battery SOC at end of day: {current_state['battery_soc']:.1f} kWh ({current_state['battery_soc']/battery_capacity*100:.1f}%)")
                zone_temp_str = ", ".join([f"{zone}: {current_state['zone_temps'][zone]:.1f}°C" for zone in zones])
                print(f"Zone temperatures at end of day: {zone_temp_str}")
            else:
                # Handle optimization failure for the day
                print(f"Optimization failed for day {day+1}.")
                
        # Save final results and create visualization
        results = finalize_compiled_results(compiled_results)
        
        # Save results to CSV - only once at the end
        csv_filename = os.path.join(output_dir, f"building_optimization_results_{days}days.csv")
        save_compiled_results_to_csv(compiled_results, csv_filename, zones, heat_sources, cooling_sources)
        
        # Create plots
        plot_filename = os.path.join(output_dir, f"building_optimization_plot_{days}days.png")
        plot_compiled_results(compiled_results, zones, heat_sources, cooling_sources, start_date, plot_filename)
    
        print("\nOptimization completed.")
        
        # Calculate summary statistics
        total_hours = days * 24
        total_cost = sum((compiled_results["grid_imported"][i] * compiled_results["energy_price"][i] / 1000)
                        for i in range(total_hours))
        avg_energy_price = sum(compiled_results["energy_price"]) / len(compiled_results["energy_price"])
        
        # Calculate PV self-consumption rate
        total_pv = sum(compiled_results["pv_generation"])
        pv_consumed = sum(compiled_results["pv_self_consumed"]) + sum(compiled_results["battery_charge_pv"])
        pv_self_consumption_rate = pv_consumed / total_pv if total_pv > 0 else 0
        
        # Calculate totals
        total_heating = sum(compiled_results["total_heating_power"])
        total_cooling = sum(compiled_results["total_cooling_power"])
        avg_external_temp = sum(compiled_results["external_temp"]) / len(compiled_results["external_temp"])
        
        print("\nOptimization Results Summary:")
        print("-" * 60)
        print(f"Simulation period: {days} days ({total_hours} hours) starting {start_date.strftime('%Y-%m-%d')}")
        print(f"Total energy cost: €{total_cost:.2f}")
        print(f"Average energy cost: €{total_cost/(total_hours):.2f} per hour")
        print(f"PV self-consumption rate: {pv_self_consumption_rate*100:.1f}%")
        print(f"Total heating energy: {total_heating:.1f} kWh")
        print(f"Total cooling energy: {total_cooling:.1f} kWh")
        print(f"Average external temperature: {avg_external_temp:.1f}°C")
        
        return results, optimizer
    
def generate_building_data_with_cooling():
    """Generate building zones, heat sources, cooling sources, and their mappings"""
    # Define building zones
    zones = {
        "Office_North": {
            "area": 300,          # m²
            "volume": 900,        # m³
            "window_area": 60,    # m²
            "R": 0.9,             # Thermal resistance (°C/kW)
            "C": 80,             # Thermal capacitance (kWh/°C)
            "initial_temp": 20    # Initial temperature (°C)
        },
        "Office_South": {
            "area": 300,
            "volume": 900,
            "window_area": 80,    # More windows on south side
            "R": 0.9,
            "C": 80,
            "initial_temp": 20
        },
        "Large_Hall": {
            "area": 200,
            "volume": 1000,       # Higher ceiling
            "window_area": 40,
            "R": 0.5,
            "C": 200,             # Higher thermal mass
            "initial_temp": 20
        },
        "Small_Hall": {
            "area": 100,
            "volume": 500,
            "window_area": 20,
            "R": 0.7,
            "C": 100,
            "initial_temp": 20
        }
    }
    
    # Define heat sources (same as before)
    heat_sources = {
        "AirToAir_HP": {
            "capacity": 80,       # kW
            "efficiency": 3.8,    # COP (nominal - will be modified by PWL)
            "min_runtime": 3      # hours
        },
        "AirToWater_HP": {
            "capacity": 130,      # kW
            "efficiency": 3.5,    # COP (nominal - will be modified by PWL)
            "min_runtime": 3      # hours
        },
        "Boiler": {
            "capacity": 250,      # kW
            "efficiency": 1.0,    # 90% efficiency
            "min_runtime": 0      # Can turn on/off immediately
        }
    }
    
    # NEW: Define cooling sources
    cooling_sources = {
        "AirToAir_AC": {
            "capacity": 85,       # kW cooling capacity
            "efficiency": 3.5,    # EER (nominal - will be modified by PWL)
            "min_runtime": 1      # hours
        },
        "AirToWater_Chiller": {
            "capacity": 140,      # kW cooling capacity
            "efficiency": 3.3,    # EER (nominal - will be modified by PWL)
            "min_runtime": 2      # hours
        }
    }
    
    # Map heat sources to zones (same as before)
    zone_source_mapping = {
        "Office_North": {
            "AirToAir_HP": 0.8,   # 80% from VRV
            "Boiler": 0.2         # 20% from boiler (backup)
        },
        "Office_South": {
            "AirToAir_HP": 0.8,   # 80% from VRV
            "Boiler": 0.2         # 20% from boiler (backup)
        },
        "Large_Hall": {
            "AirToWater_HP": 0.9, # 90% from air-water (floor heating)
            "Boiler": 0.1         # 10% from boiler (backup)
        },
        "Small_Hall": {
            "AirToWater_HP": 0.9, # 90% from air-water (fan coils)
            "Boiler": 0.1         # 10% from boiler (backup)
        }
    }
    
    # NEW: Map cooling sources to zones (similar structure to heating)
    zone_cooling_mapping = {
        "Office_North": {
            "AirToAir_AC": 1.0,   # 100% from VRF/VRV in cooling mode
        },
        "Office_South": {
            "AirToAir_AC": 1.0,   # 100% from VRF/VRV in cooling mode
        },
        "Large_Hall": {
            "AirToWater_Chiller": 1.0, # 100% from chiller (fan coils)
        },
        "Small_Hall": {
            "AirToWater_Chiller": 1.0, # 100% from chiller (fan coils)
        }
    }
    
    return zones, heat_sources, cooling_sources, zone_source_mapping, zone_cooling_mapping

def initialize_empty_results(total_hours, zones, heat_sources, cooling_sources):
    """Initialize empty storage for compiled results"""
    compiled_results = {
        "timestamps": [],
        "is_working_hour": [],
        "external_temp": [],
        "energy_price": [],
        "total_heating_power": [],
        "total_cooling_power": [],
        "grid_imported": [],
        "pv_generation": [],
        "pv_self_consumed": [],
        "pv_waste": [],
        "battery_soc": [],
        "battery_charge": [],
        "battery_discharge": [],
        "battery_charge_grid": [],
        "battery_charge_pv": []
    }
    
    # Add zone temperatures
    for zone_name in zones:
        compiled_results[f"temp_{zone_name}"] = []
        compiled_results[f"heating_mode_{zone_name}"] = []
    
    # Add heat source data
    for source_name in heat_sources:
        compiled_results[f"power_{source_name}"] = []
        compiled_results[f"energy_{source_name}"] = []
        compiled_results[f"on_{source_name}"] = []
        
        if source_name in ["AirToAir_HP", "AirToWater_HP"]:
            compiled_results[f"cop_{source_name}"] = []
            compiled_results[f"effective_cop_{source_name}"] = []
            compiled_results[f"startup_{source_name}"] = []
    
    # Add cooling source data
    for source_name in cooling_sources:
        compiled_results[f"power_{source_name}"] = []
        compiled_results[f"energy_{source_name}"] = []
        compiled_results[f"on_{source_name}"] = []
        
        if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
            compiled_results[f"eer_{source_name}"] = []
            compiled_results[f"effective_eer_{source_name}"] = []
            compiled_results[f"startup_{source_name}"] = []
    
    return compiled_results


def use_fallback_strategy(compiled_results, current_state, hour, zones, heat_sources, 
                         cooling_sources, external_temp, energy_price):
    """Apply a simple rule-based fallback strategy when optimization fails"""
    # Current time for timestamps
    current_time = compiled_results["timestamps"][-1] if compiled_results["timestamps"] else "Fallback-Start"
    try:
        dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M") + timedelta(hours=1)
        timestamp = dt.strftime("%Y-%m-%d %H:%M")
    except:
        timestamp = f"Fallback-{hour}"
    
    # Working hour determination
    is_working_hour = 8 <= (hour % 24) < 18 and not ((hour // 24) % 7 >= 5)  # Weekday 8am-6pm
    
    # Simple thermodynamics for temperature evolution
    new_temps = {}
    heating_powers = {}
    cooling_powers = {}
    zone_modes = {}
    
    for zone_name, zone_data in zones.items():
        current_temp = current_state["zone_temps"][zone_name]
        
        # Heat loss to exterior (simplified)
        temp_change = (external_temp - current_temp) / zone_data["R"] / zone_data["C"]
        
        # Determine setpoint based on occupancy
        if is_working_hour:
            setpoint = 21.5  # Middle of comfort band
        else:
            setpoint = 18.0  # Energy-saving setpoint
            
        # Determine if heating or cooling needed
        if current_temp < setpoint - 0.5:
            # Need heating
            heat_power = min(25, (setpoint - current_temp) * zone_data["C"])  # Limit to 25kW per zone
            cool_power = 0
            mode = 1  # Heating mode
            
            # Apply heating effect
            temp_change += heat_power / zone_data["C"]
            
        elif current_temp > setpoint + 0.5:
            # Need cooling
            heat_power = 0
            cool_power = min(25, (current_temp - setpoint) * zone_data["C"])  # Limit to 25kW per zone
            mode = 0  # Cooling mode
            
            # Apply cooling effect
            temp_change -= cool_power / zone_data["C"]
            
        else:
            # Temperature is in comfort range
            heat_power = 0
            cool_power = 0
            mode = 1  # Default to heating mode
        
        # Calculate new temperature
        new_temp = current_temp + temp_change
        
        # Store results
        new_temps[zone_name] = new_temp
        heating_powers[zone_name] = heat_power
        cooling_powers[zone_name] = cool_power
        zone_modes[zone_name] = mode
    
    # Update current state temperatures
    for zone_name in zones:
        current_state["zone_temps"][zone_name] = new_temps[zone_name]
    
    # Allocate heating/cooling to sources (simplified)
    total_heating = sum(heating_powers.values())
    total_cooling = sum(cooling_powers.values())
    
    source_powers = {}
    source_energies = {}
    source_on = {}
    source_cop = {}
    source_eff_cop = {}
    source_startup = {}
    
    # Allocate heating
    if total_heating > 0:
        # Use AirToAir_HP for 70% of load if below capacity
        if "AirToAir_HP" in heat_sources:
            aa_hp_alloc = min(heat_sources["AirToAir_HP"]["capacity"], 0.7 * total_heating)
            source_powers["AirToAir_HP"] = aa_hp_alloc
            source_on["AirToAir_HP"] = 1
            
            # Simple COP estimation
            cop = 3.0 if external_temp > 0 else 2.5
            source_cop["AirToAir_HP"] = cop
            source_eff_cop["AirToAir_HP"] = cop
            source_startup["AirToAir_HP"] = 0
            source_energies["AirToAir_HP"] = aa_hp_alloc / cop
        else:
            source_powers["AirToAir_HP"] = 0
            source_on["AirToAir_HP"] = 0
            source_cop["AirToAir_HP"] = 3.0
            source_eff_cop["AirToAir_HP"] = 3.0
            source_startup["AirToAir_HP"] = 0
            source_energies["AirToAir_HP"] = 0
        
        # Use AirToWater_HP for remaining load if possible
        if "AirToWater_HP" in heat_sources:
            remaining_heat = total_heating - aa_hp_alloc if "AirToAir_HP" in heat_sources else total_heating
            aw_hp_alloc = min(heat_sources["AirToWater_HP"]["capacity"], remaining_heat)
            source_powers["AirToWater_HP"] = aw_hp_alloc
            source_on["AirToWater_HP"] = 1 if aw_hp_alloc > 0 else 0
            
            # Simple COP estimation
            cop = 2.8 if external_temp > 0 else 2.3
            source_cop["AirToWater_HP"] = cop
            source_eff_cop["AirToWater_HP"] = cop
            source_startup["AirToWater_HP"] = 0
            source_energies["AirToWater_HP"] = aw_hp_alloc / cop if aw_hp_alloc > 0 else 0
        else:
            source_powers["AirToWater_HP"] = 0
            source_on["AirToWater_HP"] = 0
            source_cop["AirToWater_HP"] = 2.8
            source_eff_cop["AirToWater_HP"] = 2.8
            source_startup["AirToWater_HP"] = 0
            source_energies["AirToWater_HP"] = 0
        
        # Use Boiler for any remaining load
        if "Boiler" in heat_sources:
            remaining_heat = total_heating - source_powers.get("AirToAir_HP", 0) - source_powers.get("AirToWater_HP", 0)
            boiler_alloc = min(heat_sources["Boiler"]["capacity"], remaining_heat)
            source_powers["Boiler"] = boiler_alloc
            source_on["Boiler"] = 1 if boiler_alloc > 0 else 0
            source_energies["Boiler"] = boiler_alloc  # Efficiency of 1.0
        else:
            source_powers["Boiler"] = 0
            source_on["Boiler"] = 0
            source_energies["Boiler"] = 0
    else:
        # No heating needed
        for source_name in heat_sources:
            source_powers[source_name] = 0
            source_on[source_name] = 0
            source_energies[source_name] = 0
            if source_name in ["AirToAir_HP", "AirToWater_HP"]:
                source_cop[source_name] = 3.0
                source_eff_cop[source_name] = 3.0
                source_startup[source_name] = 0
    
    # Similar allocation for cooling sources
    cooling_source_powers = {}
    cooling_source_energies = {}
    cooling_source_on = {}
    cooling_source_eer = {}
    cooling_source_eff_eer = {}
    cooling_source_startup = {}
    
    # Cooling allocation logic similar to heating
    if total_cooling > 0:
        # Use AirToAir_AC for 70% of load if below capacity
        if "AirToAir_AC" in cooling_sources:
            aa_ac_alloc = min(cooling_sources["AirToAir_AC"]["capacity"], 0.7 * total_cooling)
            cooling_source_powers["AirToAir_AC"] = aa_ac_alloc
            cooling_source_on["AirToAir_AC"] = 1
            
            # Simple EER estimation
            eer = 3.5 if external_temp < 30 else 3.0
            cooling_source_eer["AirToAir_AC"] = eer
            cooling_source_eff_eer["AirToAir_AC"] = eer
            cooling_source_startup["AirToAir_AC"] = 0
            cooling_source_energies["AirToAir_AC"] = aa_ac_alloc / eer
        else:
            cooling_source_powers["AirToAir_AC"] = 0
            cooling_source_on["AirToAir_AC"] = 0
            cooling_source_eer["AirToAir_AC"] = 3.5
            cooling_source_eff_eer["AirToAir_AC"] = 3.5
            cooling_source_startup["AirToAir_AC"] = 0
            cooling_source_energies["AirToAir_AC"] = 0
        
        # Use AirToWater_Chiller for remaining load if possible
        if "AirToWater_Chiller" in cooling_sources:
            remaining_cool = total_cooling - aa_ac_alloc if "AirToAir_AC" in cooling_sources else total_cooling
            aw_ch_alloc = min(cooling_sources["AirToWater_Chiller"]["capacity"], remaining_cool)
            cooling_source_powers["AirToWater_Chiller"] = aw_ch_alloc
            cooling_source_on["AirToWater_Chiller"] = 1 if aw_ch_alloc > 0 else 0
            
            # Simple EER estimation
            eer = 3.3 if external_temp < 30 else 2.8
            cooling_source_eer["AirToWater_Chiller"] = eer
            cooling_source_eff_eer["AirToWater_Chiller"] = eer
            cooling_source_startup["AirToWater_Chiller"] = 0
            cooling_source_energies["AirToWater_Chiller"] = aw_ch_alloc / eer if aw_ch_alloc > 0 else 0
        else:
            cooling_source_powers["AirToWater_Chiller"] = 0
            cooling_source_on["AirToWater_Chiller"] = 0
            cooling_source_eer["AirToWater_Chiller"] = 3.3
            cooling_source_eff_eer["AirToWater_Chiller"] = 3.3
            cooling_source_startup["AirToWater_Chiller"] = 0
            cooling_source_energies["AirToWater_Chiller"] = 0
    else:
        # No cooling needed
        for source_name in cooling_sources:
            cooling_source_powers[source_name] = 0
            cooling_source_on[source_name] = 0
            cooling_source_energies[source_name] = 0
            if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
                cooling_source_eer[source_name] = 3.5
                cooling_source_eff_eer[source_name] = 3.5
                cooling_source_startup[source_name] = 0
    
    # Simple energy balance (PV, grid, battery)
    pv_generation = 0  # Could estimate from weather if needed
    electrical_demand = sum(source_energies.values()) + sum(cooling_source_energies.values())
    
    # Battery charging/discharging strategy
    if energy_price < 50:  # Low price - charge battery
        battery_discharge = 0
        battery_charge = min(60, 200 * 0.9 - current_state["battery_soc"])  # Charge rate limited by max SOC
        battery_charge_grid = battery_charge
        battery_charge_pv = 0
        grid_imported = electrical_demand + battery_charge_grid
        pv_self_consumed = 0
        pv_waste = 0
    else:  # High price - discharge battery
        battery_charge = 0
        battery_charge_grid = 0
        battery_charge_pv = 0
        battery_discharge = min(90, electrical_demand, current_state["battery_soc"] - 200 * 0.1)  # Limit by max rate, demand, and min SOC
        grid_imported = max(0, electrical_demand - battery_discharge)
        pv_self_consumed = 0
        pv_waste = 0
    
    # Update battery SOC
    new_battery_soc = current_state["battery_soc"] + 0.95 * battery_charge - battery_discharge / 0.95
    current_state["battery_soc"] = new_battery_soc
    
    # Add to compiled results
    compiled_results["timestamps"].append(timestamp)
    compiled_results["is_working_hour"].append(is_working_hour)
    compiled_results["external_temp"].append(external_temp)
    compiled_results["energy_price"].append(energy_price)
    compiled_results["total_heating_power"].append(total_heating)
    compiled_results["total_cooling_power"].append(total_cooling)
    compiled_results["grid_imported"].append(grid_imported)
    compiled_results["pv_generation"].append(pv_generation)
    compiled_results["pv_self_consumed"].append(pv_self_consumed)
    compiled_results["pv_waste"].append(pv_waste)
    compiled_results["battery_soc"].append(current_state["battery_soc"])
    compiled_results["battery_charge"].append(battery_charge)
    compiled_results["battery_discharge"].append(battery_discharge)
    compiled_results["battery_charge_grid"].append(battery_charge_grid)
    compiled_results["battery_charge_pv"].append(battery_charge_pv)
    
    # Add zone temperatures and modes
    for zone_name in zones:
        compiled_results[f"temp_{zone_name}"].append(new_temps[zone_name])
        compiled_results[f"heating_mode_{zone_name}"].append(zone_modes[zone_name])
    
    # Add heat source data
    for source_name in heat_sources:
        compiled_results[f"power_{source_name}"].append(source_powers.get(source_name, 0))
        compiled_results[f"energy_{source_name}"].append(source_energies.get(source_name, 0))
        compiled_results[f"on_{source_name}"].append(source_on.get(source_name, 0))
        
        if source_name in ["AirToAir_HP", "AirToWater_HP"]:
            compiled_results[f"cop_{source_name}"].append(source_cop.get(source_name, 3.0))
            compiled_results[f"effective_cop_{source_name}"].append(source_eff_cop.get(source_name, 3.0))
            compiled_results[f"startup_{source_name}"].append(source_startup.get(source_name, 0))
    
    # Add cooling source data
    for source_name in cooling_sources:
        compiled_results[f"power_{source_name}"].append(cooling_source_powers.get(source_name, 0))
        compiled_results[f"energy_{source_name}"].append(cooling_source_energies.get(source_name, 0))
        compiled_results[f"on_{source_name}"].append(cooling_source_on.get(source_name, 0))
        
        if source_name in ["AirToAir_AC", "AirToWater_Chiller"]:
            compiled_results[f"eer_{source_name}"].append(cooling_source_eer.get(source_name, 3.5))
            compiled_results[f"effective_eer_{source_name}"].append(cooling_source_eff_eer.get(source_name, 3.5))
            compiled_results[f"startup_{source_name}"].append(cooling_source_startup.get(source_name, 0))
    
    print(f"Applied fallback strategy for hour {hour}. New zone temperatures: " + 
          ", ".join([f"{z}: {t:.1f}°C" for z, t in new_temps.items()]))

def store_hour_results(compiled_results, day_results, source_hour_index, target_hour_index):
    """Store the results for a specific hour from day_results into compiled_results"""
    # Basic data
    compiled_results["timestamps"].append(day_results["timestamps"][source_hour_index])
    compiled_results["is_working_hour"].append(day_results["is_working_hour"][source_hour_index])
    compiled_results["external_temp"].append(day_results["temperatures"]["external"][source_hour_index])
    compiled_results["energy_price"].append(day_results["energy"]["prices"][source_hour_index])
    compiled_results["total_heating_power"].append(day_results["heating"]["total_power"][source_hour_index])
    compiled_results["total_cooling_power"].append(day_results["cooling"]["total_power"][source_hour_index])
    compiled_results["grid_imported"].append(day_results["energy"]["grid_imported"][source_hour_index])
    compiled_results["pv_generation"].append(day_results["energy"]["pv"]["generation"][source_hour_index])
    compiled_results["pv_self_consumed"].append(day_results["energy"]["pv"]["self_consumed"][source_hour_index])
    compiled_results["pv_waste"].append(day_results["energy"]["pv"]["waste"][source_hour_index])
    
    # Battery data
    compiled_results["battery_soc"].append(day_results["energy"]["battery"]["soc"][source_hour_index])
    compiled_results["battery_charge"].append(day_results["energy"]["battery"]["charge"][source_hour_index])
    compiled_results["battery_discharge"].append(day_results["energy"]["battery"]["discharge"][source_hour_index])
    compiled_results["battery_charge_grid"].append(day_results["energy"]["battery"]["charge_grid"][source_hour_index])
    compiled_results["battery_charge_pv"].append(day_results["energy"]["battery"]["charge_pv"][source_hour_index])
    
    # Zone temperatures and modes
    for zone_name in day_results["temperatures"]["zones"]:
        compiled_results[f"temp_{zone_name}"].append(day_results["temperatures"]["zones"][zone_name][source_hour_index])
        
        # Add zone mode if available
        if "zone_modes" in day_results and zone_name in day_results["zone_modes"]:
            compiled_results[f"heating_mode_{zone_name}"].append(day_results["zone_modes"][zone_name][source_hour_index])
        else:
            # Default to heating mode (1) if not specified
            compiled_results[f"heating_mode_{zone_name}"].append(1)
    
    # Heat source data
    for source_name in day_results["heating"]["sources"]:
        compiled_results[f"power_{source_name}"].append(day_results["heating"]["sources"][source_name]["power"][source_hour_index])
        compiled_results[f"energy_{source_name}"].append(day_results["heating"]["sources"][source_name]["energy"][source_hour_index])
        compiled_results[f"on_{source_name}"].append(day_results["heating"]["sources"][source_name]["on"][source_hour_index])
        
        # Add COP data if available
        if "cop" in day_results["heating"]["sources"][source_name]:
            compiled_results[f"cop_{source_name}"].append(day_results["heating"]["sources"][source_name]["cop"][source_hour_index])
            compiled_results[f"effective_cop_{source_name}"].append(day_results["heating"]["sources"][source_name]["effective_cop"][source_hour_index])
            compiled_results[f"startup_{source_name}"].append(day_results["heating"]["sources"][source_name]["startup"][source_hour_index])
    
    # Cooling source data
    for source_name in day_results["cooling"]["sources"]:
        compiled_results[f"power_{source_name}"].append(day_results["cooling"]["sources"][source_name]["power"][source_hour_index])
        compiled_results[f"energy_{source_name}"].append(day_results["cooling"]["sources"][source_name]["energy"][source_hour_index])
        compiled_results[f"on_{source_name}"].append(day_results["cooling"]["sources"][source_name]["on"][source_hour_index])
        
        # Add EER data if available
        if "eer" in day_results["cooling"]["sources"][source_name]:
            compiled_results[f"eer_{source_name}"].append(day_results["cooling"]["sources"][source_name]["eer"][source_hour_index])
            compiled_results[f"effective_eer_{source_name}"].append(day_results["cooling"]["sources"][source_name]["effective_eer"][source_hour_index])
            compiled_results[f"startup_{source_name}"].append(day_results["cooling"]["sources"][source_name]["startup"][source_hour_index])

def save_compiled_results_to_csv(compiled_results, filename, zones, heat_sources, cooling_sources):
    """Save compiled results to CSV"""
    # Convert to DataFrame and save
    df = pd.DataFrame(compiled_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def finalize_compiled_results(compiled_results):
    """Convert compiled results to standard format for consistency with non-rolling approach"""
    # This creates a results dictionary in the same format as extract_results() would produce
    results = {
        "timestamps": compiled_results["timestamps"],
        "is_working_hour": compiled_results["is_working_hour"],
        "temperatures": {
            "external": compiled_results["external_temp"],
            "zones": {}
        },
        "heating": {
            "total_power": compiled_results["total_heating_power"],
            "sources": {}
        },
        "cooling": {
            "total_power": compiled_results["total_cooling_power"],
            "sources": {}
        },
        "energy": {
            "prices": compiled_results["energy_price"],
            "grid_imported": compiled_results["grid_imported"],
            "pv": {
                "generation": compiled_results["pv_generation"],
                "self_consumed": compiled_results["pv_self_consumed"],
                "waste": compiled_results["pv_waste"],
                "self_consumption_rate": sum(compiled_results["pv_self_consumed"]) / sum(compiled_results["pv_generation"]) 
                    if sum(compiled_results["pv_generation"]) > 0 else 0
            },
            "battery": {
                "soc": compiled_results["battery_soc"],
                "charge": compiled_results["battery_charge"],
                "discharge": compiled_results["battery_discharge"],
                "charge_grid": compiled_results["battery_charge_grid"],
                "charge_pv": compiled_results["battery_charge_pv"]
            }
        },
        "zone_modes": {}
    }
    
    # Extract zone temperatures and modes
    for key in compiled_results:
        if key.startswith("temp_"):
            zone_name = key[5:]  # Remove "temp_" prefix
            results["temperatures"]["zones"][zone_name] = compiled_results[key]
        
        if key.startswith("heating_mode_"):
            zone_name = key[13:]  # Remove "heating_mode_" prefix
            if "zone_modes" not in results:
                results["zone_modes"] = {}
            results["zone_modes"][zone_name] = compiled_results[key]
    
    # Extract heat source data
    for key in compiled_results:
        if key.startswith("power_"):
            source_name = key[6:]  # Remove "power_" prefix
            
            # Check if this is a heating or cooling source
            if any(key.startswith(f"power_{name}") for name in ["AirToAir_HP", "AirToWater_HP", "Boiler"]):
                # Heating source
                if source_name not in results["heating"]["sources"]:
                    results["heating"]["sources"][source_name] = {}
                results["heating"]["sources"][source_name]["power"] = compiled_results[key]
            else:
                # Cooling source
                if source_name not in results["cooling"]["sources"]:
                    results["cooling"]["sources"][source_name] = {}
                results["cooling"]["sources"][source_name]["power"] = compiled_results[key]
        
        if key.startswith("energy_"):
            source_name = key[7:]  # Remove "energy_" prefix
            
            # Check if this is a heating or cooling source
            if any(key.startswith(f"energy_{name}") for name in ["AirToAir_HP", "AirToWater_HP", "Boiler"]):
                # Heating source
                if source_name not in results["heating"]["sources"]:
                    results["heating"]["sources"][source_name] = {}
                results["heating"]["sources"][source_name]["energy"] = compiled_results[key]
            else:
                # Cooling source
                if source_name not in results["cooling"]["sources"]:
                    results["cooling"]["sources"][source_name] = {}
                results["cooling"]["sources"][source_name]["energy"] = compiled_results[key]
        
        if key.startswith("on_"):
            source_name = key[3:]  # Remove "on_" prefix
            
            # Check if this is a heating or cooling source
            if any(key.startswith(f"on_{name}") for name in ["AirToAir_HP", "AirToWater_HP", "Boiler"]):
                # Heating source
                if source_name not in results["heating"]["sources"]:
                    results["heating"]["sources"][source_name] = {}
                results["heating"]["sources"][source_name]["on"] = compiled_results[key]
            else:
                # Cooling source
                if source_name not in results["cooling"]["sources"]:
                    results["cooling"]["sources"][source_name] = {}
                results["cooling"]["sources"][source_name]["on"] = compiled_results[key]
    
    # Special fields for heat pumps - COP, effective COP, startup
    for key in compiled_results:
        if key.startswith("cop_"):
            source_name = key[4:]  # Remove "cop_" prefix
            results["heating"]["sources"][source_name]["cop"] = compiled_results[key]
        
        if key.startswith("effective_cop_"):
            source_name = key[14:]  # Remove "effective_cop_" prefix
            results["heating"]["sources"][source_name]["effective_cop"] = compiled_results[key]
        
        if key.startswith("startup_") and any(key.startswith(f"startup_{name}") for name in ["AirToAir_HP", "AirToWater_HP"]):
            source_name = key[8:]  # Remove "startup_" prefix
            results["heating"]["sources"][source_name]["startup"] = compiled_results[key]
    
    # Special fields for cooling - EER, effective EER, startup
    for key in compiled_results:
        if key.startswith("eer_"):
            source_name = key[4:]  # Remove "eer_" prefix
            results["cooling"]["sources"][source_name]["eer"] = compiled_results[key]
        
        if key.startswith("effective_eer_"):
            source_name = key[14:]  # Remove "effective_eer_" prefix
            results["cooling"]["sources"][source_name]["effective_eer"] = compiled_results[key]
        
        if key.startswith("startup_") and any(key.startswith(f"startup_{name}") for name in ["AirToAir_AC", "AirToWater_Chiller"]):
            source_name = key[8:]  # Remove "startup_" prefix
            results["cooling"]["sources"][source_name]["startup"] = compiled_results[key]
    
    # Calculate total cost
    results["total_cost"] = sum((results["energy"]["grid_imported"][i] * results["energy"]["prices"][i] / 1000)
                               for i in range(len(results["timestamps"])))
    
    return results


def plot_compiled_results(compiled_results, zones, heat_sources, cooling_sources, start_date, save_path=None):
    """Plot compiled results from rolling horizon optimization"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    
    # Convert string timestamps to datetime objects
    try:
        datetime_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M") for ts in compiled_results["timestamps"]]
    except:
        # In case of fallback timestamps, generate from start_date
        hours = len(compiled_results["timestamps"])
        datetime_timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Create figure with subplots - 5 subplots including cooling
    fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    # Plot 1: Zone temperatures and external temperature
    ax1 = axs[0]
    ax1.plot(datetime_timestamps, compiled_results["external_temp"], 'k-', label="External", linewidth=1.5)
    
    # Plot each zone temperature
    zone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, zone_name in enumerate(zones):
        color = zone_colors[i % len(zone_colors)]
        ax1.plot(datetime_timestamps, compiled_results[f"temp_{zone_name}"], 
                 '-', color=color, label=zone_name, linewidth=2)
    
    # Add comfort ranges
    # Find contiguous blocks of working/non-working hours
    blocks = []
    current_block = {"start": 0, "is_working": compiled_results["is_working_hour"][0]}
    
    # Identify contiguous blocks
    for i in range(1, len(compiled_results["is_working_hour"])):
        if compiled_results["is_working_hour"][i] != current_block["is_working"]:
            current_block["end"] = i - 1
            blocks.append(current_block)
            current_block = {"start": i, "is_working": compiled_results["is_working_hour"][i]}
    
    # Add the last block
    current_block["end"] = len(compiled_results["is_working_hour"]) - 1
    blocks.append(current_block)
    
    # Define comfort ranges
    comfort_temp_min_working = 20.0
    comfort_temp_max_working = 23.0
    comfort_temp_min_nonworking = 16.0
    comfort_temp_max_nonworking = 25.0
    
    # Now fill_between for each contiguous block
    working_label_used = False
    nonworking_label_used = False
    
    for block in blocks:
        block_x = [datetime_timestamps[i] for i in range(block["start"], block["end"] + 1)]
        if block["is_working"]:
            label = "Working Hours Comfort Range" if not working_label_used else ""
            working_label_used = True
            ax1.fill_between(block_x, comfort_temp_min_working, comfort_temp_max_working, 
                            color='g', alpha=0.2, label=label)
        else:
            label = "Non-working Hours Comfort Range" if not nonworking_label_used else ""
            nonworking_label_used = True
            ax1.fill_between(block_x, comfort_temp_min_nonworking, comfort_temp_max_nonworking, 
                            color='b', alpha=0.15, label=label)
    
    # Add day separators
    day_boundaries = []
    current_date = datetime_timestamps[0].date()
    
    # Find day boundaries
    for i, dt in enumerate(datetime_timestamps):
        if dt.date() != current_date:
            day_boundaries.append((i, dt))
            current_date = dt.date()
    
    # Add vertical lines at midnight for each day
    for i, dt in day_boundaries:
        # Add day separator to all subplots
        for ax in axs:
            ax.axvline(x=dt, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
        # Add day label for first subplot only
        day_name = dt.strftime("%A")  # Full day name
        date_str = dt.strftime("%m-%d")
        ax1.text(dt, ax1.get_ylim()[1] * 0.98, f"{day_name}\n{date_str}", 
                horizontalalignment='left', verticalalignment='top',
                fontsize=8, color='gray', rotation=0,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.set_title("Zone Temperatures", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heat source power
    ax2 = axs[1]
    
    # Create stacked area plot for heat sources
    source_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    cumulative = np.zeros(len(datetime_timestamps))
    
    for i, source_name in enumerate(heat_sources):
        source_power = np.array(compiled_results[f"power_{source_name}"])
        color = source_colors[i % len(source_colors)]
        
        # Plot the cumulative line
        ax2.plot(datetime_timestamps, source_power + cumulative, 
                '-', color=color, label=source_name, linewidth=1.5)
        
        # Fill area between this line and previous cumulative
        ax2.fill_between(datetime_timestamps, cumulative, source_power + cumulative, 
                         color=color, alpha=0.5)
        
        # Update cumulative for next source
        cumulative += source_power
    
    # Add total power as a dashed line on top
    ax2.plot(datetime_timestamps, compiled_results["total_heating_power"], 
             'k--', label="Total Heating Power", linewidth=1.5)
    
    ax2.set_ylabel("Heating Power (kW)", fontsize=12)
    ax2.set_title("Heat Source Utilization", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cooling source power
    ax3 = axs[2]
    
    # Create stacked area plot for cooling sources
    cooling_colors = ['#17becf', '#9467bd', '#bcbd22', '#e377c2', '#7f7f7f']
    cumulative = np.zeros(len(datetime_timestamps))
    
    for i, source_name in enumerate(cooling_sources):
        source_power = np.array(compiled_results[f"power_{source_name}"])
        color = cooling_colors[i % len(cooling_colors)]
        
        # Plot the cumulative line
        ax3.plot(datetime_timestamps, source_power + cumulative, 
                '-', color=color, label=source_name, linewidth=1.5)
        
        # Fill area between this line and previous cumulative
        ax3.fill_between(datetime_timestamps, cumulative, source_power + cumulative, 
                         color=color, alpha=0.5)
        
        # Update cumulative for next source
        cumulative += source_power
    
    # Add total cooling power as a dashed line on top
    ax3.plot(datetime_timestamps, compiled_results["total_cooling_power"], 
             'k--', label="Total Cooling Power", linewidth=1.5)
    
    ax3.set_ylabel("Cooling Power (kW)", fontsize=12)
    ax3.set_title("Cooling Source Utilization", fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PV and battery
    ax4 = axs[3]
    
    # PV generation and consumption
    ax4.plot(datetime_timestamps, compiled_results["pv_generation"], 
             'y-', alpha=0.8, label="PV Generation", linewidth=2)
    ax4.plot(datetime_timestamps, compiled_results["pv_self_consumed"], 
             'g-', label="PV Self-consumed", linewidth=1.5)
    ax4.plot(datetime_timestamps, compiled_results["pv_waste"], 
             'r-', label="PV Waste", linewidth=1.5)
    
    # Grid imports
    ax4.plot(datetime_timestamps, compiled_results["grid_imported"], 
             'k-', alpha=0.7, label="Grid Import", linewidth=1.5)
    
    # Battery charge/discharge
    battery_charge = np.array(compiled_results["battery_charge"])
    battery_discharge = np.array(compiled_results["battery_discharge"])
    ax4.plot(datetime_timestamps, battery_discharge, 'c-', label="Battery Discharge", linewidth=1.5)
    ax4.plot(datetime_timestamps, -battery_charge, 'm-', label="Battery Charge", linewidth=1.5)
    
    ax4.set_ylabel("Power (kW)", fontsize=12)
    ax4.set_title("Energy System Operation", fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', bbox_to_anchor=(1.01, 0.95), borderaxespad=0)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Battery SOC and energy prices
    ax5 = axs[4]
    
    # Primary y-axis: Battery SOC
    soc_percentage = np.array(compiled_results["battery_soc"]) / 200 * 100  # Assuming 200 kWh capacity
    ax5.plot(datetime_timestamps, soc_percentage, 'b-', label="Battery SOC", linewidth=2)
    ax5.set_ylabel("Battery SOC (%)", fontsize=12)
    ax5.set_ylim(0, 100)
    
    # Secondary y-axis: Energy prices
    ax5_twin = ax5.twinx()
    
    # Plot day-ahead energy prices
    ax5_twin.plot(datetime_timestamps, compiled_results["energy_price"], 
                 'r-', label="Day-Ahead Price", linewidth=2)
    
    # Add markers to make the price trend more visible
    price_marker_interval = max(1, len(datetime_timestamps) // 20)  # Show about 20 markers
    ax5_twin.plot([datetime_timestamps[i] for i in range(0, len(datetime_timestamps), price_marker_interval)], 
                 [compiled_results["energy_price"][i] for i in range(0, len(compiled_results["energy_price"]), price_marker_interval)],
                 'ro', markersize=4)
    
    ax5_twin.set_ylabel("Energy Price (€/MWh)", fontsize=12)
    
    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    
    ax5.set_title("Battery State of Charge and Energy Prices", fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # X-axis label for the bottom subplot
    ax5.set_xlabel("Time", fontsize=12)
    
    # Format x-axis dates
    days = len(datetime_timestamps) // 24
    
    # Choose appropriate date formatting based on simulation length
    if days <= 3:
        # For shorter periods, show more detailed time
        date_fmt = mdates.DateFormatter('%m-%d %H:%M')
        # Show every 6 hours for 1-3 days
        interval = 6
    elif days <= 7:
        # For medium periods, show day and hour
        date_fmt = mdates.DateFormatter('%m-%d %H:%M')
        # Show every 12 hours for 4-7 days
        interval = 12
    else:
        # For longer periods, show only days
        date_fmt = mdates.DateFormatter('%m-%d')
        # Show every day at midnight
        interval = 24
    
    # Apply formatting to all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
        
        # Improve tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Rotate x-tick labels for better readability
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def annuity(c, k, r, L):
    """
    Calculate annuity (annual payment) for an investment
    
    Parameters:
    -----------
    c : float
        Unit cost (€/kW, €/kWh, etc.)
    k : float
        Capacity (kW, kWh, etc.)
    r : float
        Discount rate (e.g., 0.06 for 6%)
    L : int
        Asset lifetime in years
    
    Returns:
    --------
    float
        Annual payment
    """
    return 0 if k == 0 or L == 0 else c * k * (r * (1 + r) ** L) / ((1 + r) ** L - 1)

def calculate_investment_costs(capacities, discount_rate=0.06):
    """
    Calculate total annualized investment costs
    
    Parameters:
    -----------
    capacities : dict
        Dictionary with capacities for different assets:
        {
            "battery_capacity": float,  # kWh
            "battery_power": float,     # kW
            "air_to_air_hp": float,     # kW
            "air_to_water_hp": float,   # kW
            "thermal_R": float,         # Improvement in R-value
            "thermal_C": float          # Improvement in C-value
        }
    discount_rate : float
        Discount rate
    
    Returns:
    --------
    dict
        Annual costs for each component and total
    """
    # Unit costs (€/unit)
    unit_costs = {
        "battery_capacity": 300,    # €/kWh
        "battery_power": 400,       # €/kW
        "air_to_air_hp": 1200,      # €/kW
        "air_to_water_hp": 1400,    # €/kW
        "thermal_R": 5000,          # €/unit R improvement
        "thermal_C": 200            # €/kWh/°C
    }
    
    # Asset lifetimes (years)
    lifetimes = {
        "battery": 10,
        "heat_pump": 20,
        "thermal": 30
    }
    
    # O&M costs (% of investment)
    om_rates = {
        "battery": 0.02,      # 2% per year
        "heat_pump": 0.03,    # 3% per year
        "thermal": 0.005      # 0.5% per year
    }
    
    # Calculate annuities for each component
    battery_cap_annuity = annuity(
        unit_costs["battery_capacity"], 
        capacities.get("battery_capacity", 0),
        discount_rate,
        lifetimes["battery"]
    )
    
    battery_pow_annuity = annuity(
        unit_costs["battery_power"], 
        capacities.get("battery_power", 0),
        discount_rate,
        lifetimes["battery"]
    )
    
    air_air_hp_annuity = annuity(
        unit_costs["air_to_air_hp"], 
        capacities.get("air_to_air_hp", 0),
        discount_rate,
        lifetimes["heat_pump"]
    )
    
    air_water_hp_annuity = annuity(
        unit_costs["air_to_water_hp"], 
        capacities.get("air_to_water_hp", 0),
        discount_rate,
        lifetimes["heat_pump"]
    )
    
    thermal_R_annuity = annuity(
        unit_costs["thermal_R"], 
        capacities.get("thermal_R", 0),
        discount_rate,
        lifetimes["thermal"]
    )
    
    thermal_C_annuity = annuity(
        unit_costs["thermal_C"], 
        capacities.get("thermal_C", 0),
        discount_rate,
        lifetimes["thermal"]
    )
    
    # Calculate O&M costs
    battery_inv = (unit_costs["battery_capacity"] * capacities.get("battery_capacity", 0) + 
                  unit_costs["battery_power"] * capacities.get("battery_power", 0))
    heat_pump_inv = (unit_costs["air_to_air_hp"] * capacities.get("air_to_air_hp", 0) + 
                    unit_costs["air_to_water_hp"] * capacities.get("air_to_water_hp", 0))
    thermal_inv = (unit_costs["thermal_R"] * capacities.get("thermal_R", 0) + 
                  unit_costs["thermal_C"] * capacities.get("thermal_C", 0))
    
    battery_om = battery_inv * om_rates["battery"]
    heat_pump_om = heat_pump_inv * om_rates["heat_pump"]
    thermal_om = thermal_inv * om_rates["thermal"]
    
    # Calculate total costs
    total_battery_cost = battery_cap_annuity + battery_pow_annuity + battery_om
    total_heat_pump_cost = air_air_hp_annuity + air_water_hp_annuity + heat_pump_om
    total_thermal_cost = thermal_R_annuity + thermal_C_annuity + thermal_om
    
    total_annual_cost = total_battery_cost + total_heat_pump_cost + total_thermal_cost
    
    return {
        "battery": {
            "capacity_annuity": battery_cap_annuity,
            "power_annuity": battery_pow_annuity,
            "om_cost": battery_om,
            "total": total_battery_cost
        },
        "heat_pump": {
            "air_to_air_annuity": air_air_hp_annuity,
            "air_to_water_annuity": air_water_hp_annuity,
            "om_cost": heat_pump_om,
            "total": total_heat_pump_cost
        },
        "thermal": {
            "R_value_annuity": thermal_R_annuity,
            "C_value_annuity": thermal_C_annuity,
            "om_cost": thermal_om,
            "total": total_thermal_cost
        },
        "total_annual_cost": total_annual_cost
    }

def analyze_investment(capacities, optimization_results, simulation_days=9):
    """
    Analyze investment based on a single optimization run
    
    Parameters:
    -----------
    capacities : dict
        Dictionary with capacities for different assets
    optimization_results : dict
        Results from optimization
    simulation_days : int
        Number of days in the simulation
        
    Returns:
    --------
    dict
        Investment analysis results
    """
    # Calculate annual investment costs
    investment_costs = calculate_investment_costs(capacities)
    
    # Calculate annual operational costs
    days_per_year = 365
    scaling_factor = days_per_year / simulation_days
    
    # Extract total cost from results and scale to annual
    simulation_cost = optimization_results["total_cost"] 
    annual_operational_cost = simulation_cost * scaling_factor
    
    # Calculate simple payback period (years)
    total_investment = (
        capacities.get("battery_capacity", 0) * 300 + 
        capacities.get("battery_power", 0) * 400 +
        capacities.get("air_to_air_hp", 0) * 1200 + 
        capacities.get("air_to_water_hp", 0) * 1400 +
        capacities.get("thermal_R", 0) * 5000 + 
        capacities.get("thermal_C", 0) * 200
    )
    
    # Generate results
    results = {
        "capacities": capacities,
        "annual_investment_costs": investment_costs,
        "annual_operational_cost": annual_operational_cost,
        "total_annual_cost": annual_operational_cost + investment_costs["total_annual_cost"],
        "total_investment": total_investment,
        "operational_cost_per_day": simulation_cost / simulation_days,
        "simulation_days": simulation_days
    }
    
    return results

def save_investment_analysis(results, filename="investment_analysis_results.json"):
    """Save investment analysis results to JSON file"""
    import json
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Investment analysis saved to: {filename}")

def run_investment_analysis(days=9, output_dir="./results", start_date=datetime(2023, 1, 13, 0, 0)):
    """
    Run optimization and perform investment analysis
    
    Parameters:
    -----------
    days : int
        Number of days to simulate
    output_dir : str
        Directory to save results
    start_date : datetime
        Start date for simulation
        
    Returns:
    --------
    dict
        Investment analysis results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define system capacities
    capacities = {
        "battery_capacity": 300,   # kWh
        "battery_power": 90,       # kW
        "air_to_air_hp": 80,       # kW
        "air_to_water_hp": 130,    # kW
        "thermal_R": 0,            # No thermal envelope improvement
        "thermal_C": 0             # No thermal mass improvement
    }
    
    # Run the optimization
    print(f"Running optimization for {days} days...")
    results, optimizer = run_building_optimization_with_cooling(
        days=days,
        output_dir=output_dir,
        start_date=start_date,
        rolling_horizon=True,
        forecast_hours=24
    )
    
    # Print heat source usage summary
    print_heat_source_usage_summary(results)
    
    # Perform investment analysis
    print("\nPerforming investment analysis...")
    analysis = analyze_investment(
        capacities, 
        results,
        simulation_days=days
    )
    
    # Save analysis
    analysis_filename = os.path.join(output_dir, "investment_analysis.json")
    save_investment_analysis(analysis, analysis_filename)
    
    # Print summary of investment analysis
    print("\nInvestment Analysis Summary:")
    print("-" * 60)
    print(f"Annual operational cost: €{analysis['annual_operational_cost']:.2f}")
    print(f"Annual investment costs: €{analysis['annual_investment_costs']['total_annual_cost']:.2f}")
    print(f"Total annual cost: €{analysis['total_annual_cost']:.2f}")
    print(f"Total investment: €{analysis['total_investment']:.2f}")
    
    # Break down by component
    print("\nInvestment Cost Breakdown:")
    print(f"  Battery: €{analysis['annual_investment_costs']['battery']['total']:.2f}/year")
    print(f"  Heat Pumps: €{analysis['annual_investment_costs']['heat_pump']['total']:.2f}/year")
    print(f"  Thermal Improvements: €{analysis['annual_investment_costs']['thermal']['total']:.2f}/year")
    
    print(f"\nInvestment analysis complete. Results saved to {analysis_filename}")
    return analysis

def print_heat_source_usage_summary(results):
    """Print a summary of heat source usage in kWh"""
    print("\nHeat Source Usage Summary:")
    print("-" * 60)
    
    # Extract heat source usage from results
    total_hours = len(results["timestamps"])
    
    # Calculate total energy per heat source (kWh)
    heat_source_energy = {}
    for source_name in results["heating"]["sources"]:
        # Sum of power (kW) over hours = energy (kWh)
        source_data = results["heating"]["sources"][source_name]
        energy_used = sum(source_data["energy"])
        heat_generated = sum(source_data["power"])
        
        # Store in summary dict
        heat_source_energy[source_name] = {
            "electricity_consumed": energy_used,
            "heat_generated": heat_generated
        }
        
        # Calculate average COP if available
        if "cop" in source_data:
            # Filter to only include hours when the source was active
            active_hours = sum(1 for on_val in source_data["on"] if on_val > 0.5)
            if active_hours > 0:
                avg_cop = sum(cop for cop, on_val in zip(source_data["cop"], source_data["on"]) 
                             if on_val > 0.5) / active_hours
                heat_source_energy[source_name]["average_cop"] = avg_cop
    
    # Print summary
    total_electricity = 0
    total_heat = 0
    
    for source_name, data in heat_source_energy.items():
        electricity = data["electricity_consumed"]
        heat = data["heat_generated"]
        total_electricity += electricity
        total_heat += heat
        
        print(f"{source_name}:")
        print(f"  Electricity consumed: {electricity:.1f} kWh")
        print(f"  Heat generated: {heat:.1f} kWh")
        
        if "average_cop" in data:
            print(f"  Average COP when active: {data['average_cop']:.2f}")
        
        # Calculate percentage of total heat
        if total_heat > 0:
            heat_percentage = (heat / sum(results["heating"]["total_power"])) * 100
            print(f"  Contribution to total heating: {heat_percentage:.1f}%")
        
        print()
    
    # Print totals
    print("Total Summary:")
    print(f"  Total electricity for heating: {total_electricity:.1f} kWh")
    print(f"  Total heat generated: {total_heat:.1f} kWh")
    if total_electricity > 0:
        print(f"  Overall system COP: {total_heat/total_electricity:.2f}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run building optimization model')
    parser.add_argument('--days', type=int, default=9, help='Number of days to simulate')
    parser.add_argument('--start-date', type=str, default='2023-01-13', help='Start date in format YYYY-MM-DD')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--investment-analysis', action='store_true', help='Run investment analysis')
    args = parser.parse_args()
    
    # Convert start date string to datetime
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        print(f"Invalid start date format: {args.start_date}. Please use YYYY-MM-DD.")
        start_date = datetime(2023, 1, 13, 0, 0)
        print(f"Using default start date: {start_date.strftime('%Y-%m-%d')}")
    
    if args.investment_analysis:
        # Run optimization with investment analysis
        run_investment_analysis(
            days=args.days,
            output_dir=args.output_dir,
            start_date=start_date
        )
    else:
        # Run the standard optimization
        results, optimizer = run_building_optimization_with_cooling(
            days=args.days,
            output_dir=args.output_dir,
            start_date=start_date,
            rolling_horizon=True,
            forecast_hours=24
        )
        # Print heat source usage summary
        print_heat_source_usage_summary(results)
        