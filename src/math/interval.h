#ifndef INTERVAL_H
#define INTERVAL_H

class Interval {
    public:
        float min, max;

        Interval() : min(+std::numeric_limits<float>::infinity()), max(-std::numeric_limits<float>::infinity()) {}
        Interval(float min, float max) : min(min), max(max) {}

        float size() const { return max - min; }
        bool contains(float value) const { return value >= min && value <= max; }
        bool surrounds(const Interval &other) const { return min <= other.min && max >= other.max; }

        float clamp(float value) const {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        static const Interval empty, universe;
};

const Interval Interval::empty = Interval(+std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
const Interval Interval::universe = Interval(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

Interval operator+(const Interval& ival, double displacement) {
    return Interval(ival.min + displacement, ival.max + displacement);
}

Interval operator+(double displacement, const Interval& ival) {
    return ival + displacement;
}

#endif // INTERVAL_H