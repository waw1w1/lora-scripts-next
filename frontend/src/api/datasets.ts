import { apiData } from "./client"

export interface DatasetsRoot { root: string; default: string; exists: boolean }
export interface DatasetOverview {
  state: "computing" | "ready" | "error"
  file_count: number | null
  captioned_count: number | null
  total_bytes: number | null
  updated_at: string | null
  computed_at?: string | null
  error?: string | null
}
export interface DatasetEntry { name: string; path: string; overview: DatasetOverview | null }
export interface DatasetList { root: string; exists: boolean; datasets: DatasetEntry[] }
export interface DatasetCreated { name: string; path: string }

export const datasetsApi = {
  getRoot: () => apiData<DatasetsRoot>("/api/datasets/root"),
  updateRoot: (path: string) => apiData<DatasetsRoot>("/api/datasets/root", { method: "PUT", body: JSON.stringify({ path }) }),
  list: () => apiData<DatasetList>("/api/datasets"),
  create: (name: string) => apiData<DatasetCreated>("/api/datasets", { method: "POST", body: JSON.stringify({ name }) }),
  overview: (name: string) => apiData<{ name: string; overview: DatasetOverview }>(`/api/datasets/${encodeURIComponent(name)}/overview`),
}
