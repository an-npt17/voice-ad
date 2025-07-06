{
  pkgs ? import <nixpkgs> { },
}:

pkgs.stdenv.mkDerivation {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    gcc12
    portaudio # Here it is!
  ];
  shellHook = ''
    export CC=${pkgs.gcc12}/bin/gcc
    export CXX=${pkgs.gcc12}/bin/g++
    export PATH=${pkgs.gcc12}/bin:$PATH
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
