{
  description = "uv2nix devshell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        workspaceRoot = ./.;
        venvName = "venv";
        python = pkgs.python311;

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = workspaceRoot; };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        baseSet = pkgs.callPackage inputs.pyproject-nix.build.packages {
          inherit python;
        };
        pythonSet = baseSet.overrideScope (
          pkgs.lib.composeManyExtensions [
            inputs.pyproject-build-systems.overlays.default
            overlay
          ]
        );
        venv = pythonSet.mkVirtualEnv "${venvName}" workspace.deps.default;
      in
      {
        packages.default = pkgs.writeShellApplication {
          name = "run-silero-vad";
          runtimeInputs = [
            pkgs.uv
            venv
            pkgs.portaudio
          ];
          text = ''
            export LD_LIBRARY_PATH="${pkgs.portaudio}/lib:${pkgs.stdenv.cc.libc}/lib"
            uv run silero-vad.py
          '';
        };
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.uv
            venv
            pkgs.portaudio
          ];
          LD_LIBRARY_PATH = "${pkgs.portaudio}/lib:${pkgs.stdenv.cc.libc}/lib";
        };
      }
    );
}
