export function dxFromN(n: number): number {
    return 1 / (n - 1);
  }
  
  export function tauFromMu(mu: number, alpha: number, dx: number): number {
    return (mu * dx * dx) / alpha;
  }
  
  export function expectedK(mu: number, s: number): number {
    return Math.ceil((4 * mu) / s);
  }  