import os
import site

site_packages = site.getsitepackages()[0]
total_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(site_packages)
    for f in files
)

print(f"Total size of installed packages: {total_size / (1024 * 1024):.2f} MB")
