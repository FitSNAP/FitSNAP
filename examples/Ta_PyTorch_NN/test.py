import sysconfig

abi_flags = sysconfig.get_config_var('abiflags')

print(len(abi_flags))