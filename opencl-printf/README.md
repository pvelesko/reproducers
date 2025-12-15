# IGC SPIR-V printf bug

## Bug
`clLinkProgram` fails when SPIR-V contains `OpExtInst printf` with a UniformConstant pointer as `%s` argument.

## Error
```
<origin>: error: Invalid record (Producer: 'LLVM16.0.6' Reader: 'LLVM 16.0.6')
```

## Reproduce
```bash
make run      # FAILS - printf("%s", uniformconstant_ptr)
make run-good # WORKS - printf without pointer args
```

## Files
- `strings_bad.spv` - Contains `OpExtInst printf %fmt %uniformconstant_str` → FAILS
- `dynamic_good.spv` - Contains `OpExtInst printf %fmt %integer` → WORKS

## Environment
- Intel Arc A770
- Driver 25.44.36015.5
