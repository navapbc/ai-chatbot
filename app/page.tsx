'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 h-16">
        <div className="flex items-center justify-between h-full px-6">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-black">ASP</h1>
          </div>
          <div className="flex items-center space-x-4">
            <Link href="/login">
              <Button variant="outline" size="sm" className="border-[#cac4d0] rounded-full">
                Log In
              </Button>
            </Link>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="flex-1">
        <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-8 py-16">
          <div className="max-w-2xl w-full text-center">
            {/* Welcome Message */}
            <h1 className="text-5xl font-bold text-black mb-8 leading-tight">
              Welcome
            </h1>
            
            {/* Description */}
            <p className="text-xl text-gray-600 mb-12 leading-relaxed">
              Get started with our AI-powered benefit application assistant
            </p>
            
            {/* Consent Page Button */}
            <div className="flex justify-center">
              <Link href="/consent">
                <Button 
                  size="lg"
                  className="px-8 py-4 text-lg font-medium bg-[#b14092] hover:bg-[#9a3579] text-white rounded-2xl transition-colors duration-200"
                >
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
