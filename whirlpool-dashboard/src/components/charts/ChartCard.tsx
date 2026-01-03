import React from 'react';

interface ChartCardProps {
    title: React.ReactNode;
    children: React.ReactNode;
    className?: string;
    headerRight?: React.ReactNode;
}

export const ChartCard: React.FC<ChartCardProps> = ({ title, children, className = '', headerRight }) => {
    return (
        <div className={`bg-gray-800/50 rounded-xl p-6 border border-gray-700/50 backdrop-blur-sm ${className}`}>
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-semibold text-gray-100 flex items-center gap-3">{title}</h3>
                {headerRight && <div>{headerRight}</div>}
            </div>
            <div className="w-full h-[300px]">
                {children}
            </div>
        </div>
    );
};
