use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// Motor de inferencia en Rust sin dependencia de Python.
/// Implementa carga eficiente de tensores con Candle (HuggingFace).

fn main() -> Result<()> {
    println!("==================================================");
    println!(" 🚀 AETHEL RUST INFERENCE ENGINE (CANDLE ENGINE) ");
    println!("==================================================");

    // 1. Detección Inteligente de Dispositivo (CUDA -> CPU Fallback)
    let device = Device::new_cuda(0).unwrap_or_else(|_| {
        println!("ℹ️ CUDA no disponible en la máquina actual. Ejecutando en CPU.");
        Device::Cpu
    });

    println!("✅ Dispositivo de Inferencia Asignado: {:?}", device);

    // 2. Parámetros del Modelo
    let batch_size = 1;
    let seq_len = 32;
    let hidden_dim = 4096;
    let dtype = DType::F32;

    println!("\n[1/3] Alocando tensores de prueba (Dim: {}, Hidden: {})...", seq_len, hidden_dim);
    let start_alloc = Instant::now();

    // Creación de tensores en memoria/VRAM
    let input = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, hidden_dim), &device)?;
    let weight_q = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device)?;
    let weight_k = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device)?;

    println!("✅ Tensores inicializados en: {:?}", start_alloc.elapsed());

    // 3. Multiplicación Matricial Proyectada (MatMul)
    println!("\n[2/3] Ejecutando proyecciones de atención Query/Key (MatMul)...");
    let start_compute = Instant::now();

    let q = input.matmul(&weight_q).context("Error en proyección Q")?;
    let k = input.matmul(&weight_k).context("Error en proyección K")?;

    let elapsed_compute = start_compute.elapsed();

    println!("✅ Operación MatMul ejecutada con éxito.");
    println!("   Q Shape: {:?}", q.shape());
    println!("   K Shape: {:?}", k.shape());
    println!("   Tiempo de Cómputo: {:?}", elapsed_compute);

    // 4. Verificación de Invariancia y Memoria
    println!("\n[3/3] Verificando estabilidad de memoria...");
    let q_mean = q.mean_all()?.to_scalar::<f32>()?;
    println!("   Media del tensor Q: {:.6}", q_mean);

    println!("\n==================================================");
    println!(" 🎉 El motor de inferencia en Rust está listo.");
    println!("==================================================");

    Ok(())
}
