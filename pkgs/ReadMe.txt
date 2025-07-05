# Fix: Installation error: "libgcrypt.so.11: cannot open shared object file".
## Source: https://github.com/openai/mujoco-py/issues/323#issuecomment-618365770

Commands:
$mkdir pkgs && cd pkgs
$curl -o https://rpmfind.net/linux/atrpms/sl6-x86_64/atrpms/testing/libgcrypt11-1.4.0-15.el6.x86_64.rpm
$rpm2cpio libgcrypt11-1.4.0-15.el6.x86_64.rpm | cpio -id
$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/pkgs/usr/lib64