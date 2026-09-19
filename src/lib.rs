#![feature(clamp_to)]

pub mod callconv;

#[cfg(feature = "x86")]
pub mod x86;

#[cfg(feature = "x86_64")]
pub mod x86_64;

#[cfg(feature = "skyarch")]
pub mod skyarch;

#[cfg(feature = "w65")]
pub mod w65;

pub mod target;

pub mod helpers;

pub mod xva;

pub mod layout;
