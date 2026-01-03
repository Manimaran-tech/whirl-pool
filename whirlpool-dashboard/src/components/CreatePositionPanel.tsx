import { useState, useEffect, useCallback } from 'react';
import type { FC } from 'react';
import { X, Loader2, Minus, Plus, ChevronLeft, Settings, Info, AlertTriangle } from 'lucide-react';
import { getTokenPrice } from '../services/priceService';
import { useWallet, useConnection } from '@solana/wallet-adapter-react';
import { api } from '../api';
import { deserializeTransaction } from '../utils/transactions';
import { PriceChart } from './charts/PriceChart';
import { getCoinGeckoId } from '../utils/coinMapping';
import { MLInsightsPanel } from './MLInsightsPanel';
import { TokenNewsPanel } from './TokenNewsPanel';





interface CreatePositionPanelProps {
    isOpen: boolean;
    onClose: () => void;
    poolAddress: string;
    tokenA?: string;
    tokenB?: string;
}

type ViewMode = 'deposit' | 'range';
type RangePreset = '1%' | '5%' | '10%' | 'custom';

export const CreatePositionPanel: FC<CreatePositionPanelProps> = ({
    isOpen,
    onClose,
    poolAddress,
    tokenA = 'SOL',
    tokenB = 'USDC'
}) => {
    const { publicKey, signTransaction, connected } = useWallet();
    const { connection } = useConnection();

    // View state
    const [viewMode, setViewMode] = useState<ViewMode>('deposit');

    const [currentPrice, setCurrentPrice] = useState<number>(0);
    const [tokenAPriceUsd, setTokenAPriceUsd] = useState<number>(0);
    const [tokenBPriceUsd, setTokenBPriceUsd] = useState<number>(0);
    const [displayToken, setDisplayToken] = useState<string>(tokenA); // Which token to display in chart
    const [minPrice, setMinPrice] = useState<string>('');
    const [maxPrice, setMaxPrice] = useState<string>('');

    // Range preset state
    const [selectedPreset, setSelectedPreset] = useState<RangePreset>('5%');

    // Deposit state
    const [amountA, setAmountA] = useState<string>('');
    const [amountB, setAmountB] = useState<string>('');
    const slippage = 1; // 1% default slippage

    // Loading and transaction states
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [priceLoading, setPriceLoading] = useState(true);
    const [txStatus, setTxStatus] = useState<'idle' | 'building' | 'signing' | 'confirming' | 'success' | 'error'>('idle');
    const [txSignature, setTxSignature] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);


    // Fetch current price on mount
    useEffect(() => {
        if (!isOpen || !poolAddress) return;

        const fetchPrice = async () => {
            setPriceLoading(true);
            try {
                console.log("CreatePositionPanel: Fetching prices for", tokenA, tokenB);

                // 1. Fetch USD prices for both tokens (using reliable priceService)
                const priceA = await getTokenPrice(tokenA);
                const priceB = await getTokenPrice(tokenB);

                console.log("CreatePositionPanel: USD Prices:", { [tokenA]: priceA, [tokenB]: priceB });

                setTokenAPriceUsd(priceA);
                setTokenBPriceUsd(priceB);

                // 2. Determine which token to display for yield farming
                // Priority: Show the ALTCOIN (not SOL, not stablecoin)
                // - SOL/PENGU → show PENGU (altcoin to yield farm)
                // - JupSOL/SOL → show JupSOL (altcoin to yield farm)
                // - SOL/USDC → show SOL (no altcoin, so show SOL not stablecoin)
                const stablecoins = ['USDC', 'USDT'];
                const isTokenAStable = stablecoins.includes(tokenA);
                const isTokenBStable = stablecoins.includes(tokenB);
                const isTokenASOL = tokenA === 'SOL';
                const isTokenBSOL = tokenB === 'SOL';

                let displayTokenA: boolean;

                if (isTokenAStable) {
                    // TokenA is stablecoin → show tokenB (SOL or altcoin)
                    displayTokenA = false;
                } else if (isTokenBStable) {
                    // TokenB is stablecoin → show tokenA (SOL or altcoin)
                    displayTokenA = true;
                } else if (isTokenASOL && !isTokenBSOL) {
                    // SOL/Altcoin → show Altcoin (tokenB)
                    displayTokenA = false;
                } else if (isTokenBSOL && !isTokenASOL) {
                    // Altcoin/SOL → show Altcoin (tokenA)
                    displayTokenA = true;
                } else {
                    // Both are altcoins or both are SOL → show tokenA
                    displayTokenA = true;
                }

                const displayPrice = displayTokenA ? priceA : priceB;
                const displayToken = displayTokenA ? tokenA : tokenB;

                console.log('CreatePositionPanel: Display token:', displayToken, '=', displayPrice);

                setDisplayToken(displayToken);

                if (displayPrice > 0) {
                    setCurrentPrice(displayPrice);

                    // Apply default preset immediately based on this USD price
                    const percentage = 0.05; // Default 5%
                    const min = displayPrice * (1 - percentage);
                    const max = displayPrice * (1 + percentage);
                    setMinPrice(min.toFixed(4));
                    setMaxPrice(max.toFixed(4));
                } else {
                    console.warn("CreatePositionPanel: Failed to fetch USD price for", displayToken);
                }

            } catch (error) {
                console.error("CreatePositionPanel: Price fetch error:", error);
            } finally {
                setPriceLoading(false);
            }
        };

        fetchPrice();
        const interval = setInterval(fetchPrice, 30000); // Poll every 30s
        return () => clearInterval(interval);

    }, [isOpen, poolAddress, tokenA, tokenB]);

    // Reset states when opening
    useEffect(() => {
        if (isOpen) {
            setTxStatus('idle');
            setErrorMessage(null);
            setTxSignature(null);
        }
    }, [isOpen]);

    // Apply range preset
    const applyPreset = useCallback((preset: RangePreset, price: number) => {
        if (price <= 0) return;

        let percentage = 0;
        switch (preset) {
            case '1%': percentage = 0.01; break;
            case '5%': percentage = 0.05; break;
            case '10%': percentage = 0.10; break;
            default: return; // Custom - don't auto-set
        }

        const min = price * (1 - percentage);
        const max = price * (1 + percentage);

        setMinPrice(min.toFixed(4));
        setMaxPrice(max.toFixed(4));
        setSelectedPreset(preset);
    }, []);

    // Calculate deposit ratio based on current price and range
    const calculateDepositRatio = useCallback((): { ratioA: number; ratioB: number } => {
        if (!minPrice || !maxPrice || currentPrice <= 0) {
            return { ratioA: 50, ratioB: 50 };
        }

        const min = parseFloat(minPrice);
        const max = parseFloat(maxPrice);

        if (currentPrice <= min) {
            return { ratioA: 0, ratioB: 100 };
        } else if (currentPrice >= max) {
            return { ratioA: 100, ratioB: 0 };
        } else {
            const rangePosition = (currentPrice - min) / (max - min);
            const ratioB = Math.round(rangePosition * 100 * 10) / 10;
            return { ratioA: 100 - ratioB, ratioB };
        }
    }, [minPrice, maxPrice, currentPrice]);

    const { ratioA, ratioB } = calculateDepositRatio();

    // Auto-calculate the other token amount based on the deposit ratio
    const handleAmountAChange = useCallback((value: string) => {
        setAmountA(value);
        if (value && ratioA > 0 && ratioB > 0 && currentPrice > 0) {
            // Calculate equivalent amount of tokenB based on ratio
            // For a 50/50 split, if price is 1:1, amounts should be equal
            // The ratio tells us the proportion of each token in the position
            const numValue = parseFloat(value);
            if (!isNaN(numValue)) {
                // Convert based on ratio: if ratioA=50, ratioB=50, amounts are proportional
                // amountB = amountA * (ratioB / ratioA) * priceRatio
                // For JupSOL/SOL pair, price ~1, so it's roughly 1:1 at 50/50
                const calculatedB = numValue * (ratioB / ratioA) * currentPrice;
                setAmountB(calculatedB.toFixed(9));
            }
        } else if (!value) {
            setAmountB('');
        }
    }, [ratioA, ratioB, currentPrice]);

    const handleAmountBChange = useCallback((value: string) => {
        setAmountB(value);
        if (value && ratioA > 0 && ratioB > 0 && currentPrice > 0) {
            const numValue = parseFloat(value);
            if (!isNaN(numValue)) {
                // Reverse calculation
                const calculatedA = numValue * (ratioA / ratioB) / currentPrice;
                setAmountA(calculatedA.toFixed(9));
            }
        } else if (!value) {
            setAmountA('');
        }
    }, [ratioA, ratioB, currentPrice]);

    // Check if in range
    const isInRange = currentPrice >= parseFloat(minPrice || '0') && currentPrice <= parseFloat(maxPrice || '0');

    const handleCreatePosition = async () => {
        if (!publicKey || !signTransaction) {
            setErrorMessage("Please connect your wallet first.");
            return;
        }

        if (!amountA || !minPrice || !maxPrice) {
            setErrorMessage("Please enter all required fields.");
            return;
        }

        setIsSubmitting(true);
        setTxStatus('building');
        setErrorMessage(null);

        try {
            console.log("Creating position:", {
                poolAddress,
                minPrice,
                maxPrice,
                amountA,
                slippage
            });

            // Convert prices from Display (USD) to Pool Units (TokenB / TokenA)
            // If Token B is NOT a stablecoin (e.g. PENGU), we must divide USD price by Token B USD price.
            // Example: Lower = $126 (USD/SOL) / $0.01 (USD/PENGU) = 12,600 PENGU/SOL
            let submissionLower = minPrice;
            let submissionUpper = maxPrice;

            if (tokenBPriceUsd > 0 && !['USDC', 'USDT'].includes(tokenB)) {
                submissionLower = (parseFloat(minPrice) / tokenBPriceUsd).toFixed(6);
                submissionUpper = (parseFloat(maxPrice) / tokenBPriceUsd).toFixed(6);
                console.log(`Converting USD bounds to Pool Units: ${minPrice} -> ${submissionLower}, ${maxPrice} -> ${submissionUpper}`);
            }

            const response = await api.createOrDeposit({
                wallet: publicKey.toString(),
                whirlpool: poolAddress,
                priceLower: submissionLower,
                priceUpper: submissionUpper,
                amountA: amountA
            });

            if (!response.success || !response.serializedTransaction) {
                throw new Error(response.error || "Failed to build transaction");
            }

            // Deserialize transaction
            const transaction = deserializeTransaction(response.serializedTransaction);

            setTxStatus('signing');
            console.log("Building openPosition transaction...");
            console.log("Requesting wallet signature...");

            const signedTx = await signTransaction(transaction);
            const signature = await connection.sendRawTransaction(signedTx.serialize());

            setTxSignature(signature);

            setTxStatus('confirming');
            console.log("Transaction sent:", signature);

            await connection.confirmTransaction(signature, 'confirmed');

            // Success
            setTxStatus('success');

        } catch (error) {
            console.error("Position creation failed:", error);
            setTxStatus('error');
            setErrorMessage((error as Error).message || "Transaction failed. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const getStatusMessage = () => {
        switch (txStatus) {
            case 'building': return 'Building transaction...';
            case 'signing': return 'Please approve in your wallet...';
            case 'confirming': return 'Confirming on-chain...';
            case 'success': return 'Position created successfully!';
            case 'error': return 'Transaction failed';
            default: return '';
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <div className="bg-card w-full max-w-7xl rounded-2xl border border-border shadow-2xl animate-in fade-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-border sticky top-0 bg-card z-10">
                    <div className="flex items-center gap-2">
                        {viewMode === 'range' && (
                            <button
                                onClick={() => setViewMode('deposit')}
                                className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                                <ChevronLeft size={20} />
                            </button>
                        )}
                        <h3 className="text-lg font-bold">Create Position</h3>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {viewMode === 'deposit' ? (
                    /* Deposit View - 4 Column Layout: News | Chart | Inputs | AI */
                    <div className="flex flex-col lg:flex-row h-full">
                        {/* Column 1: News Panel */}
                        <div className="w-full lg:w-[15%] p-4 border-b lg:border-b-0 lg:border-r border-border bg-muted/10 overflow-hidden">
                            <TokenNewsPanel
                                tokenA={tokenA}
                                tokenB={tokenB}
                                isOpen={isOpen}
                            />
                        </div>

                        {/* Column 2: Chart (Wider) */}
                        <div className="w-full lg:w-[35%] p-4 border-b lg:border-b-0 lg:border-r border-border bg-muted/10 space-y-4 overflow-hidden flex flex-col">
                            <div className="flex-1 w-full min-h-[350px]">
                                <PriceChart
                                    coinId={getCoinGeckoId(displayToken)}
                                    title={`${displayToken} Price`}
                                />
                            </div>
                        </div>

                        {/* Column 3: Inputs */}
                        <div className="w-full lg:w-[25%] p-4 space-y-4 border-b lg:border-b-0 lg:border-r border-border">
                            {/* Info Banner */}
                            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 flex items-start gap-2">
                                <Info className="text-blue-400 shrink-0 mt-0.5" size={16} />
                                <p className="text-xs text-blue-200">
                                    Fees are earned only while the price is within your selected range.
                                </p>
                            </div>

                            {/* Range Presets */}
                            <div className="flex items-center gap-2">
                                {(['1%', '5%', '10%', 'custom'] as RangePreset[]).map((preset) => (
                                    <button
                                        key={preset}
                                        onClick={() => {
                                            if (preset === 'custom') {
                                                setSelectedPreset('custom');
                                                setViewMode('range');
                                            } else {
                                                applyPreset(preset, currentPrice);
                                            }
                                        }}
                                        className={`flex - 1 py - 2 px - 3 rounded - lg text - sm font - medium transition - all ${selectedPreset === preset
                                            ? 'bg-primary text-primary-foreground'
                                            : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                                            } `}
                                    >
                                        {preset === 'custom' ? 'Custom' : `±${preset} `}
                                    </button>
                                ))}
                            </div>

                            {/* Range Display */}
                            <div className="bg-muted/30 rounded-xl p-4 space-y-3">
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Price Range</span>
                                    <span className="font-mono">
                                        ${minPrice || '—'} - ${maxPrice || '—'}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Current Price</span>
                                    <span className="font-mono font-medium">
                                        {priceLoading ? 'Loading...' : `$${currentPrice.toFixed(4)}`}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                    <span className="text-muted-foreground">Deposit Ratio</span>
                                    <span className="font-medium">
                                        {ratioA.toFixed(0)}% {tokenA} / {ratioB.toFixed(0)}% {tokenB}
                                    </span>
                                </div>

                                {/* Range Status */}
                                {!isInRange && minPrice && maxPrice && (
                                    <div className="flex items-center gap-2 text-yellow-400 text-xs pt-2 border-t border-border/50">
                                        <AlertTriangle size={14} />
                                        <span>Current price is outside your range</span>
                                    </div>
                                )}
                            </div>

                            {/* Deposit Amounts */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium">Deposit Amount</span>
                                </div>

                                {/* Token A Input */}
                                <div className="bg-background border border-border rounded-xl p-4">
                                    <div className="flex items-center justify-between">
                                        <input
                                            type="number"
                                            value={amountA}
                                            onChange={(e) => handleAmountAChange(e.target.value)}
                                            placeholder="0"
                                            className="bg-transparent text-2xl font-medium focus:outline-none w-full"
                                        />
                                        <div className="flex items-center gap-2 ml-2">
                                            <div className="w-6 h-6 rounded-full bg-gradient-to-r from-purple-500 to-blue-500"></div>
                                            <span className="font-bold">{tokenA}</span>
                                        </div>
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">
                                        ${amountA ? (parseFloat(amountA) * (tokenAPriceUsd || currentPrice)).toFixed(2) : '0.00'}
                                    </div>
                                </div>

                                {/* Token B Input (auto-calculated based on ratio) */}
                                <div className="bg-background border border-border rounded-xl p-4">
                                    <div className="flex items-center justify-between">
                                        <input
                                            type="number"
                                            value={amountB}
                                            onChange={(e) => handleAmountBChange(e.target.value)}
                                            placeholder="0"
                                            className="bg-transparent text-2xl font-medium focus:outline-none w-full"
                                        />
                                        <div className="flex items-center gap-2 ml-2">
                                            <div className="w-6 h-6 rounded-full bg-green-500"></div>
                                            <span className="font-bold">{tokenB}</span>
                                        </div>
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">
                                        ${amountB ? (parseFloat(amountB) * (tokenBPriceUsd || 1)).toFixed(2) : '0.00'}
                                    </div>
                                </div>
                            </div>

                            {/* Settings Row */}
                            <div className="flex items-center justify-between pt-2">
                                <button
                                    onClick={() => setViewMode('range')}
                                    className="text-sm text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
                                >
                                    <Settings size={14} />
                                    Adjust Range
                                </button>
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-muted-foreground">Slippage:</span>
                                    <span className="text-xs font-medium bg-muted/50 px-2 py-1 rounded">
                                        {slippage}%
                                    </span>
                                </div>
                            </div>

                            {/* Transaction Status */}
                            {txStatus !== 'idle' && (
                                <div className={`p - 4 rounded - lg border ${txStatus === 'success'
                                    ? 'bg-green-500/10 border-green-500/30'
                                    : txStatus === 'error'
                                        ? 'bg-red-500/10 border-red-500/30'
                                        : 'bg-blue-500/10 border-blue-500/30'
                                    } `}>
                                    <div className="flex items-center gap-2">
                                        {txStatus !== 'success' && txStatus !== 'error' && (
                                            <Loader2 className="animate-spin" size={16} />
                                        )}
                                        <span className={`text - sm ${txStatus === 'success' ? 'text-green-400' : txStatus === 'error' ? 'text-red-400' : 'text-blue-400'} `}>
                                            {getStatusMessage()}
                                        </span>
                                    </div>
                                    {txSignature && (
                                        <a
                                            href={`https://solscan.io/tx/${txSignature}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-xs text-primary hover:underline mt-2 block"
                                        >
                                            View on Solscan →
                                        </a >
                                    )}
                                    {
                                        errorMessage && (
                                            <p className="text-xs text-red-400 mt-2">{errorMessage}</p>
                                        )
                                    }
                                </div >
                            )}

                            {/* On-chain Transaction Notice */}
                            <div className="text-xs text-muted-foreground">
                                ⚡ This is an on-chain transaction requiring SOL for gas fees.
                            </div>

                            {/* Create Position Button */}
                            {
                                txStatus === 'success' ? (
                                    <button
                                        onClick={onClose}
                                        className="w-full py-4 bg-secondary text-secondary-foreground font-bold rounded-xl hover:bg-secondary/80 transition-colors"
                                    >
                                        Close
                                    </button>
                                ) : (
                                    <button
                                        onClick={handleCreatePosition}
                                        disabled={isSubmitting || !connected}
                                        className="w-full py-4 bg-primary text-primary-foreground font-bold rounded-xl hover:bg-primary/90 transition-all shadow-lg shadow-primary/20 flex items-center justify-center gap-2 disabled:opacity-50"
                                    >
                                        {isSubmitting && <Loader2 className="animate-spin" size={20} />}
                                        {!connected ? 'Connect Wallet' : isSubmitting ? 'Creating...' : 'Create Position'}
                                    </button>
                                )
                            }
                        </div>

                        {/* Column 4: AI Insights */}
                        <div className="w-full lg:w-[25%] p-4">
                            <MLInsightsPanel
                                tokenA={tokenA}
                                tokenB={tokenB}
                                isOpen={isOpen}
                                currentPriceA={
                                    // If B is stable (e.g. USDC), then Pool Price (B per A) is the price of A in USD
                                    (['USDC', 'USDT'].includes(tokenB) && currentPrice > 0)
                                        ? currentPrice
                                        : (tokenAPriceUsd || undefined)
                                }
                                currentPriceB={
                                    // If A is stable (e.g. USDC), then Pool Price (B per A) is Price of A in B. 
                                    // Implies Price of B in A = 1/Pool Price.
                                    (['USDC', 'USDT'].includes(tokenA) && currentPrice > 0)
                                        ? (1 / currentPrice)
                                        : (tokenBPriceUsd || undefined)
                                }
                                onPredictedRangeChange={(lower, upper) => {
                                    // ML returns lower/upper in USD for Token A.
                                    // Since we are now in "USD Mode" for display, we just use them directly.
                                    if (selectedPreset !== 'custom') {
                                        setMinPrice(lower.toFixed(4));
                                        setMaxPrice(upper.toFixed(4));
                                    }
                                }}
                            />
                        </div>
                    </div>
                ) : (
                    /* Range View */
                    <div className="p-4 space-y-4">
                        {/* Position Range Header */}
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-medium">Position Range</span>
                            <span className="text-xs text-muted-foreground">
                                USD per {tokenA}
                            </span>
                        </div>

                        {/* Full/Custom Toggle */}
                        <div className="flex bg-muted/30 rounded-lg p-1">
                            <button
                                onClick={() => {
                                    setMinPrice('0');
                                    setMaxPrice('999999');
                                }}
                                className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${minPrice === '0' ? 'bg-background shadow' : ''
                                    }`}
                            >
                                Full Range
                            </button>
                            <button
                                onClick={() => applyPreset('5%', currentPrice)}
                                className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${minPrice !== '0' ? 'bg-background shadow' : ''
                                    }`}
                            >
                                Custom
                            </button>
                        </div>

                        {/* Visual Range Selector */}
                        <div className="bg-muted/20 rounded-xl p-4 h-32 relative">
                            {priceLoading ? (
                                <div className="flex items-center justify-center h-full">
                                    <Loader2 className="animate-spin text-muted-foreground" size={24} />
                                </div>
                            ) : (
                                <div className="h-full flex items-end justify-center gap-1">
                                    {Array.from({ length: 20 }).map((_, i) => {
                                        const min = parseFloat(minPrice) || 0;
                                        const max = parseFloat(maxPrice) || 0;
                                        const step = currentPrice * 0.01;
                                        const barPrice = currentPrice - (10 - i) * step;
                                        const isInRange = barPrice >= min && barPrice <= max;
                                        const isCurrentPrice = Math.abs(barPrice - currentPrice) < step / 2;

                                        return (
                                            <div
                                                key={i}
                                                className={`w-2 rounded-t transition-all ${isCurrentPrice
                                                    ? 'bg-yellow-400'
                                                    : isInRange
                                                        ? 'bg-primary/70'
                                                        : 'bg-muted/50'
                                                    }`}
                                                style={{
                                                    height: `${20 + Math.random() * 60}%`,
                                                }}
                                            />
                                        );
                                    })}
                                </div>
                            )}

                            {/* Range indicators */}
                            <div className="absolute bottom-0 left-4 text-xs text-red-400">
                                -{((1 - parseFloat(minPrice) / currentPrice) * 100).toFixed(1) || '0'}%
                            </div>
                            <div className="absolute bottom-0 right-4 text-xs text-green-400">
                                +{((parseFloat(maxPrice) / currentPrice - 1) * 100).toFixed(1) || '0'}%
                            </div>
                        </div>

                        {/* Min/Max Price Inputs */}
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-xs text-muted-foreground">Min Price</label>
                                <div className="flex items-center bg-background border border-border rounded-lg">
                                    <button
                                        onClick={() => setMinPrice((parseFloat(minPrice) - 1).toFixed(4))}
                                        className="p-2 text-muted-foreground hover:text-foreground"
                                    >
                                        <Minus size={16} />
                                    </button>
                                    <input
                                        type="number"
                                        value={minPrice}
                                        onChange={(e) => {
                                            setMinPrice(e.target.value);
                                            setSelectedPreset('custom');
                                        }}
                                        className="flex-1 bg-transparent text-center font-mono text-sm focus:outline-none"
                                    />
                                    <button
                                        onClick={() => setMinPrice((parseFloat(minPrice) - 1).toFixed(4))}
                                        className="p-2 text-muted-foreground hover:text-foreground"
                                    >
                                        <Minus size={16} />
                                    </button>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs text-muted-foreground">Max Price</label>
                                <div className="flex items-center bg-background border border-border rounded-lg">
                                    <button
                                        onClick={() => setMaxPrice((parseFloat(maxPrice) - 1).toFixed(4))}
                                        className="p-2 text-muted-foreground hover:text-foreground"
                                    >
                                        <Minus size={16} />
                                    </button>
                                    <input
                                        type="number"
                                        value={maxPrice}
                                        onChange={(e) => {
                                            setMaxPrice(e.target.value);
                                            setSelectedPreset('custom');
                                        }}
                                        className="flex-1 bg-transparent text-center font-mono text-sm focus:outline-none"
                                    />
                                    <button
                                        onClick={() => setMaxPrice((parseFloat(maxPrice) + 1).toFixed(4))}
                                        className="p-2 text-muted-foreground hover:text-foreground"
                                    >
                                        <Plus size={16} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Slippage */}
                        <div className="flex items-center justify-between pt-2">
                            <div className="w-8"></div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">Slippage:</span>
                                <span className="text-xs font-medium bg-muted/50 px-2 py-1 rounded">
                                    {slippage}%
                                </span>
                            </div>
                        </div>

                        {/* Confirm Button */}
                        <button
                            onClick={() => setViewMode('deposit')}
                            className="w-full py-4 bg-primary text-primary-foreground font-bold rounded-xl hover:bg-primary/90 transition-all"
                        >
                            Confirm Range
                        </button>
                    </div>
                )}
            </div >
        </div >
    );
};
