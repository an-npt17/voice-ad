{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenv.mkDerivation {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    gcc
    portaudio
    uv
    cmake
  ];
  shellHook = ''
    export CC=${pkgs.gcc}/bin/gcc
    export CXX=${pkgs.gcc}/bin/g++
    export PATH=${pkgs.gcc}/bin:$PATH
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
