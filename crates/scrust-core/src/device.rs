use candle_core::Device;

use crate::error::{Error, Result};

/// Which device an algorithm should run on.
///
/// Every algorithm is written once against candle tensors and takes a `Device`,
/// so the CPU path is the same code as the GPU path and doubles as the
/// correctness oracle in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceKind {
    /// Apple GPU if present, CPU otherwise.
    #[default]
    Auto,
    Gpu,
    Cpu,
}

impl DeviceKind {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "auto" => Ok(DeviceKind::Auto),
            "gpu" | "metal" => Ok(DeviceKind::Gpu),
            "cpu" => Ok(DeviceKind::Cpu),
            other => Err(Error::parameter("device", "one of auto, gpu, cpu", other)),
        }
    }

    pub fn resolve(self) -> Result<Device> {
        match self {
            DeviceKind::Cpu => Ok(Device::Cpu),
            DeviceKind::Gpu => metal_device().ok_or(Error::NoGpu),
            DeviceKind::Auto => Ok(metal_device().unwrap_or(Device::Cpu)),
        }
    }
}

/// The first Metal device, or `None` when the machine has no usable GPU.
///
/// candle's `Device::new_metal(0)` does not return an error on a machine without a
/// Metal device: it indexes an empty device list and panics
/// (`swap_remove index (is 0) should be < len (is 0)`), which is what GitHub's hosted
/// macOS runners do. The panic is caught here so that `auto` really does fall back to
/// the CPU and every GPU test can skip itself instead of failing the suite.
pub fn metal_device() -> Option<Device> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let device = std::panic::catch_unwind(|| Device::new_metal(0).ok())
        .ok()
        .flatten();
    std::panic::set_hook(previous);
    device
}

/// True when this machine can run the Metal backend.
pub fn gpu_available() -> bool {
    metal_device().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_names() {
        assert_eq!(DeviceKind::parse("auto").unwrap(), DeviceKind::Auto);
        assert_eq!(DeviceKind::parse("gpu").unwrap(), DeviceKind::Gpu);
        assert_eq!(DeviceKind::parse("metal").unwrap(), DeviceKind::Gpu);
        assert_eq!(DeviceKind::parse("cpu").unwrap(), DeviceKind::Cpu);
    }

    #[test]
    fn rejects_unknown_names() {
        assert!(DeviceKind::parse("cuda").is_err());
    }

    #[test]
    fn cpu_always_resolves() {
        assert!(matches!(DeviceKind::Cpu.resolve().unwrap(), Device::Cpu));
    }

    #[test]
    fn auto_never_fails() {
        assert!(DeviceKind::Auto.resolve().is_ok());
    }
}
