import { SearchHelper } from "../lib/search-pool.js";

// The page hands each helper a port to the arena worker.
globalThis.onmessage = (event) => {
  if (event.data?.type === "connect") new SearchHelper(event.data.port);
};
