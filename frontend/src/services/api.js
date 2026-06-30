/**
 * api.js — API client for the Gait ML Platform backend.
 * All fetch calls to FastAPI endpoints are centralised here.
 */

const BASE_URL = '/api';

async function fetchJSON(url) {
  const res = await fetch(`${BASE_URL}${url}`);
  if (!res.ok) {
    throw new Error(`API Error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

/* ── Models ──────────────────────────────────────────────────────────────── */

export function getModels() {
  return fetchJSON('/models');
}

export function getModelMetrics(modelName, dataset = 'autism') {
  return fetchJSON(`/models/${modelName}/metrics?dataset=${dataset}`);
}

export function getModelFeatures(modelName, dataset = 'autism') {
  return fetchJSON(`/models/${modelName}/features?dataset=${dataset}`);
}

export function getModelFigures(modelName) {
  return fetchJSON(`/models/${modelName}/figures`);
}

/* ── Comparison ──────────────────────────────────────────────────────────── */

export function getComparison(dataset = 'autism') {
  return fetchJSON(`/comparison?dataset=${dataset}`);
}

/* ── Datasets & Experiments ──────────────────────────────────────────────── */

export function getDatasets() {
  return fetchJSON('/datasets');
}

export function getExperiments() {
  return fetchJSON('/experiments');
}

/* ── Figure URL helper ───────────────────────────────────────────────────── */

export function getFigureURL(model, dataset, category, filename) {
  return `${BASE_URL}/figures/${model}/${dataset}/${category}/${filename}`;
}

/* Agent */

export async function chatWithAgent(message, dataset = null) {
  const res = await fetch(`${BASE_URL}/agent/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, dataset }),
  });
  if (!res.ok) {
    throw new Error(`API Error ${res.status}: ${res.statusText}`);
  }
  return res.json();
}