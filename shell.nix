{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenv.mkDerivation {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    gcc13
    portaudio
    uv
    cmake
  ];
  shellHook = ''
    export CC=${pkgs.gcc13}/bin/gcc
    export CXX=${pkgs.gcc13}/bin/g++
    export PATH=${pkgs.gcc13}/bin:$PATH
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
