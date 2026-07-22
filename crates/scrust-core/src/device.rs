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
            DeviceKind::Gpu => Device::new_metal(0).map_err(|_| Error::NoGpu),
            DeviceKind::Auto => Ok(Device::new_metal(0).unwrap_or(Device::Cpu)),
        }
    }
}

/// True when this machine can run the Metal backend.
pub fn gpu_available() -> bool {
    Device::new_metal(0).is_ok()
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
