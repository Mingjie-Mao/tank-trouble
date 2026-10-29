declare module "*?worker" {
  const WorkerConstructor: {
    new (options?: WorkerOptions): Worker;
  };
  export default WorkerConstructor;
}

declare module "*?url" {
  const url: string;
  export default url;
}
