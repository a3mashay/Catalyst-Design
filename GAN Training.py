
# GAN Training: Update Discriminator
real_loss = criterion_gan(discriminator(real_data), real_labels)
fake_loss = criterion_gan(discriminator(fake_features.detach()), fake_labels)
d_loss = real_loss + fake_loss

optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()

# GAN Training: Update Generator
g_loss = criterion_gan(discriminator(fake_features), real_labels)

optimizer_G.zero_grad()
g_loss.backward()
optimizer_G.step()
