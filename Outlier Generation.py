# Generate Latent Vectors
latent_vectors = boundary_factor * torch.randn(num_samples, noise_dim)

# Generate Outlier Features
generator.eval()  # Set Generator to evaluation mode
with torch.no_grad():
    outlier_features = generator(latent_vectors).numpy()

# Map Features to Materials
outlier_materials = [vector_to_material(vec) for vec in outlier_features]
