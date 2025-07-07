{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    { nixpkgs, ... }:
    let
      eachSystem =
        f:
        nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = eachSystem (pkgs: {
        default = pkgs.mkShell {
          buildInputs = with pkgs; [
            gcc12
            portaudio
            uv
            cmake
            python3
            basedpyright
          ];
          shellHook = ''
            export CC=${pkgs.gcc12}/bin/gcc
            export CXX=${pkgs.gcc12}/bin/g++
            export PATH=${pkgs.gcc12}/bin:$PATH
            export EXTRA_CCFLAGS="-I/usr/include"
          '';
        };
      });
      packages = eachSystem (pkgs: {
        default = pkgs.hello;
      });
    };
}
