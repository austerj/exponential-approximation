% minimax coefficients computed via chebfun package
% run in MATLAB r2024a Update 3 (24.1.0.2605623), online version

% https://matlab.mathworks.com/open/fileexchange/v1?id=47023 
addpath('FileExchange/chebfun-current-version-5.6.0.0/')

% configure chebfun on range-reduced domain
domain = [-log(2)/2, log(2)/2];
f = chebfun(@(x) exp(x), domain);
max_order = 6;
TOL = 1e-16;  % NOTE: minimax internally caps tolerance at machine precision

% format chebfun polynomial coefficients as tuple[float]
function out = coeffs2str(coeffs)
    out = sprintf("(%s),", sprintf(char(34) + "%.16f" + char(34) + ",", coeffs));
end

% print dict of order |-> tuple of coefficients
disp("MINIMAX_POLY = {")
for n = 1:max_order
    P = minimax(f, n, 'tol', TOL, 'silent');
    disp("   " + n + ": " + coeffs2str(poly(P)));
end
disp("}")

% print dict of order |-> tuple of tuples of coefficients
disp("MINIMAX_RATIONAL = {")
for n = 1:max_order
    [P, Q] = minimax(f, n, n, 'tol', TOL, 'silent');
    P_coeffs = poly(P);
    Q_coeffs = poly(Q);
    % scale highest-order P coefficient to 1
    scale = min(P_coeffs(1));
    disp("   " + n + ": (" + coeffs2str(P_coeffs ./ scale) + " " + coeffs2str(Q_coeffs ./ scale) + "),");
end
disp("}")
