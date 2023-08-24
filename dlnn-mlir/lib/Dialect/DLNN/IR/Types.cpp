/// Implements the DLNN dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/IR/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

/// Parses a list of size-like dimensions.
///
/// ENBF: `'[' ( '?' | dim-size )* ']'`
static ParseResult parseSizes(AsmParser &p, ShapeBuilder &sizes)
{
    sizes.clear();

    if (p.parseOptionalLSquare()) return success();
    while (true) {
        if (!p.parseOptionalQuestion()) {
            sizes.push_back(dynamic_size);
            continue;
        }

        if (p.parseInteger(sizes.emplace_back())) return failure();
        if (p.parseOptionalComma()) break;
    }
    if (p.parseRSquare()) return failure();

    return success();
}
/// Prints a list of size-like dimensions.
///
/// See parseSizes() for more information.
static void printSizes(AsmPrinter &p, ShapeRef sizes)
{
    p << '[';
    llvm::interleaveComma(sizes, p, [&](dim_size_t dim) {
        if (dim == dynamic_size)
            p << '?';
        else
            p << dim;
    });
    p << ']';
}

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dlnn-mlir/Dialect/DLNN/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FeatureMapType
//===----------------------------------------------------------------------===//

namespace mlir::dlnn::detail {

struct FeatureMapTypeStorage : TypeStorage {
    using KeyTy = std::tuple<FeatureMapDomain, ScalarType, AffineMap>;

    static KeyTy getKey(
        unsigned numChannels,
        ScalarType scalarType,
        ShapeRef sizes,
        AffineMap organization)
    {
        return KeyTy(
            FeatureMapDomain(numChannels, sizes),
            scalarType,
            organization);
    }

    static llvm::hash_code hashKey(const KeyTy &key)
    {
        return llvm::hash_value(key);
    }

    static FeatureMapTypeStorage*
    construct(TypeStorageAllocator &allocator, const KeyTy &key)
    {
        return new (allocator.allocate<FeatureMapTypeStorage>())
            FeatureMapTypeStorage(
                FeatureMapDomain(
                    std::get<0>(key).getNumChannels(),
                    allocator.copyInto(std::get<0>(key).getSizes())),
                std::get<1>(key),
                std::get<2>(key));
    }

    FeatureMapTypeStorage(
        FeatureMapDomain domain,
        ScalarType scalarType,
        AffineMap organization)
            : domain(domain),
              scalarType(scalarType),
              organization(organization)
    {}

    bool operator==(const KeyTy &key) const
    {
        return key == KeyTy(domain, scalarType, organization);
    }

    FeatureMapDomain domain;
    ScalarType scalarType;
    AffineMap organization;
    mutable RankedTensorType cachedTensorType;
};

} // namespace mlir::dlnn::detail

const FeatureMapDomain &FeatureMapType::getDomain() const
{
    return getImpl()->domain;
}

unsigned FeatureMapType::getNumChannels() const
{
    return getImpl()->domain.getNumChannels();
}

ScalarType FeatureMapType::getScalarType() const
{
    return getImpl()->scalarType;
}

ShapeRef FeatureMapType::getSizes() const
{
    return getImpl()->domain.getSizes();
}

AffineMap FeatureMapType::getOrganization() const
{
    return getImpl()->organization;
}

RankedTensorType FeatureMapType::getTensorType() const
{
    // NOTE: There may be data races here, but they don't matter. computed is
    //       pure and will be the same for everyone.

    if (auto cached = getImpl()->cachedTensorType) return cached;

    // NOTE: This type was verified on construction, so this cannot fail.
    auto computedShape = computeTensorShape(
                             []() { return InFlightDiagnostic(); },
                             getImpl()->domain,
                             getImpl()->organization)
                             .getValue();
    auto computed = RankedTensorType::get(computedShape, getImpl()->scalarType);
    getImpl()->cachedTensorType = computed;
    return computed;
}

LogicalResult FeatureMapType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    unsigned numChannels,
    ScalarType scalarType,
    ArrayRef<dim_size_t> sizes,
    AffineMap organization)
{
    if (failed(FeatureMapDomain::verify(emitError, numChannels, sizes)))
        return failure();

    if (!scalarType) return emitError() << "expected scalar type";

    // NOTE: This will fail if the map is invalid. Sadly, we can't cache the
    //       result yet, because we don't have an instance.
    return computeTensorShape(
        emitError,
        FeatureMapDomain(numChannels, sizes),
        organization);
}

Type FeatureMapType::parse(AsmParser &p)
{
    // '<'
    if (p.parseLess()) return Type{};

    // (^$numChannels 'x')?
    unsigned numChannels = 1;
    auto maybeNumChannels = p.parseOptionalInteger(numChannels);
    if (maybeNumChannels.hasValue()) {
        if (maybeNumChannels.getValue()) return Type{};
        if (p.parseXInDimensionList()) return Type{};
    }

    // $scalarType
    Type scalarType;
    if (p.parseType(scalarType)) return Type{};

    // sizes
    Shape sizes;
    if (parseSizes(p, sizes)) return Type{};

    // (`,` $organization^)?
    AffineMap organization;
    if (!p.parseOptionalComma()) {
        AffineMapAttr attr;
        if (p.parseCustomAttributeWithFallback<AffineMapAttr>(attr))
            return Type{};
        organization = attr.getAffineMap();
    } else {
        organization =
            getDefaultOrganization(p.getContext(), 1 + sizes.size(), true);
    }

    // '>'
    if (p.parseGreater()) return Type{};

    return getChecked(
        p.getEncodedSourceLoc(p.getNameLoc()),
        p.getContext(),
        numChannels,
        scalarType.dyn_cast<ScalarType>(),
        sizes,
        organization);
}

void FeatureMapType::print(AsmPrinter &p) const
{
    p << '<';

    if (getNumChannels() != 1) p << getNumChannels() << 'x';

    p.printType(getScalarType());

    printSizes(p, getSizes());

    if (!isDefaultOrganization()) {
        p << ", ";
        p.printStrippedAttrOrType(AffineMapAttr::get(getOrganization()));
    }

    p << '>';
}

//===----------------------------------------------------------------------===//
// WeightsType
//===----------------------------------------------------------------------===//

namespace mlir::dlnn::detail {

struct WeightsTypeStorage : TypeStorage {
    using KeyTy = std::tuple<WeightsDomain, ScalarType, AffineMap>;

    static KeyTy getKey(
        unsigned numInChannels,
        unsigned numOutChannels,
        ScalarType scalarType,
        ShapeRef sizes,
        AffineMap organization)
    {
        return KeyTy(
            WeightsDomain(numInChannels, numOutChannels, sizes),
            scalarType,
            organization);
    }

    static llvm::hash_code hashKey(const KeyTy &key)
    {
        return llvm::hash_value(key);
    }

    static WeightsTypeStorage*
    construct(TypeStorageAllocator &allocator, const KeyTy &key)
    {
        return new (allocator.allocate<WeightsTypeStorage>())
            WeightsTypeStorage(
                WeightsDomain(
                    std::get<0>(key).getNumInChannels(),
                    std::get<0>(key).getNumOutChannels(),
                    allocator.copyInto(std::get<0>(key).getSizes())),
                std::get<1>(key),
                std::get<2>(key));
    }

    WeightsTypeStorage(
        WeightsDomain domain,
        ScalarType scalarType,
        AffineMap organization)
            : domain(domain),
              scalarType(scalarType),
              organization(organization)
    {}

    bool operator==(const KeyTy &key) const
    {
        return key == KeyTy(domain, scalarType, organization);
    }

    WeightsDomain domain;
    ScalarType scalarType;
    AffineMap organization;
    mutable RankedTensorType cachedTensorType;
};

} // namespace mlir::dlnn::detail

const WeightsDomain &WeightsType::getDomain() const
{
    return getImpl()->domain;
}

unsigned WeightsType::getNumInChannels() const
{
    return getImpl()->domain.getNumInChannels();
}

unsigned WeightsType::getNumOutChannels() const
{
    return getImpl()->domain.getNumOutChannels();
}

ScalarType WeightsType::getScalarType() const { return getImpl()->scalarType; }

ShapeRef WeightsType::getSizes() const { return getImpl()->domain.getSizes(); }

AffineMap WeightsType::getOrganization() const
{
    return getImpl()->organization;
}

RankedTensorType WeightsType::getTensorType() const
{
    // NOTE: There may be data races here, but they don't matter. computed is
    //       pure and will be the same for everyone.

    if (auto cached = getImpl()->cachedTensorType) return cached;

    // NOTE: This type was verified on construction, so this cannot fail.
    auto computedShape = computeTensorShape(
                             []() { return InFlightDiagnostic(); },
                             getImpl()->domain,
                             getImpl()->organization)
                             .getValue();
    auto computed = RankedTensorType::get(computedShape, getImpl()->scalarType);
    getImpl()->cachedTensorType = computed;
    return computed;
}

LogicalResult WeightsType::verify(
    function_ref<InFlightDiagnostic()> emitError,
    unsigned numInChannels,
    unsigned numOutChannels,
    ScalarType scalarType,
    ArrayRef<dim_size_t> sizes,
    AffineMap organization)
{
    if (failed(WeightsDomain::verify(
            emitError,
            numInChannels,
            numOutChannels,
            sizes)))
        return failure();

    if (!scalarType) return emitError() << "expected scalar type";

    // NOTE: This will fail if the map is invalid. Sadly, we can't cache the
    //       result yet, because we don't have an instance.
    return computeTensorShape(
        emitError,
        WeightsDomain(numInChannels, numOutChannels, sizes),
        organization);
}

Type WeightsType::parse(AsmParser &p)
{
    // '<'
    if (p.parseLess()) return Type{};

    // ($numInChannels^ '->')?
    unsigned numInChannels = 1;
    Shape sizes;
    dim_size_t temp;
    auto maybeFirst = p.parseOptionalInteger(temp);
    if (maybeFirst.hasValue()) {
        if (maybeFirst.getValue()) return Type{};
        if (!p.parseOptionalArrow())
            numInChannels = temp;
        else {
            sizes.push_back(temp);
            if (p.parseXInDimensionList()) return Type{};
        }
    }

    // `(dim-size 'x')*`
    while (true) {
        auto maybeNext = p.parseOptionalInteger(temp);
        if (!maybeNext.hasValue()) break;
        sizes.push_back(temp);
        if (maybeNext.getValue()) return Type{};
        if (p.parseXInDimensionList()) return Type{};
    }

    // $scalarType
    Type scalarType;
    if (p.parseType(scalarType)) return Type{};

    // ('->' $numOutChannels^)?
    unsigned numOutChannels = 1;
    if (!p.parseOptionalArrow()) {
        if (p.parseInteger(numOutChannels)) return Type{};
    }

    // (`,` $organization^)?
    AffineMap organization;
    if (!p.parseOptionalComma()) {
        AffineMapAttr attr;
        if (p.parseCustomAttributeWithFallback<AffineMapAttr>(attr))
            return Type{};
        organization = attr.getAffineMap();
    } else {
        organization =
            getDefaultOrganization(p.getContext(), 2 + sizes.size(), false);
    }

    // '>'
    if (p.parseGreater()) return Type{};

    return getChecked(
        p.getEncodedSourceLoc(p.getNameLoc()),
        p.getContext(),
        numInChannels,
        numOutChannels,
        scalarType.dyn_cast<ScalarType>(),
        sizes,
        organization);
}

void WeightsType::print(AsmPrinter &p) const
{
    p << '<';

    if (getNumInChannels() != 1) p << getNumInChannels() << "->";

    if (!getSizes().empty()) {
        llvm::interleave(getSizes(), p, "x");
        p << 'x';
    }
    p.printType(getScalarType());

    if (getNumOutChannels() != 1) p << "->" << getNumOutChannels();

    if (!isDefaultOrganization()) {
        p << ", ";
        p.printStrippedAttrOrType(AffineMapAttr::get(getOrganization()));
    }

    p << '>';
}

//===----------------------------------------------------------------------===//
// DLNNDialect
//===----------------------------------------------------------------------===//

void DLNNDialect::registerTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "dlnn-mlir/Dialect/DLNN/IR/Types.cpp.inc"
        >();
}

Type DLNNDialect::parseType(DialectAsmParser &parser) const
{
    StringRef typeTag;
    if (failed(parser.parseKeyword(&typeTag))) return Type();

    Type genType;
    auto parseResult = generatedTypeParser(parser, typeTag, genType);
    if (parseResult.hasValue()) return genType;

    parser.emitError(parser.getNameLoc(), "unknown dlnn type: ") << typeTag;
    return Type();
}

void DLNNDialect::printType(Type type, DialectAsmPrinter &printer) const
{
    if (failed(generatedTypePrinter(type, printer)))
        llvm_unreachable("unexpected dlnn type kind");
}
