from level1_steane import level_1_steane

# Generate two rounds of Steane stabilizer measurements.
circuit = level_1_steane(2)

print("Circuit generated successfully.")
print("Qubits:", circuit.num_qubits)
print("Detectors:", circuit.num_detectors)
print("Observables:", circuit.num_observables)

# Test whether Stim can construct a detector error model.
try:
    dem = circuit.detector_error_model()
    print("Detector error model generated successfully.")
except Exception as e:
    print("Detector error model error:")
    print(e)
